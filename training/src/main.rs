#![recursion_limit = "256"]

use burn::backend::{cuda::Cuda, cuda::CudaDevice, Autodiff};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::checkpoint::MetricCheckpointingStrategy;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use bytemuck::{cast_slice, pod_read_unaligned};
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;

// Import your CNN model
use burn_ai_model::new_cnn::{BattleCnn, BattleCnnConfig};

// --- CONSTANTS ---
// Must match the data generation script!
const GRID_SIZE: usize = 11;
const CHANNELS: usize = 8;
const META_FEATS: usize = 3;

// Record layout: [board: 8*11*11][meta: 3][label: 1] = 968 + 3 + 1 = 972 floats
const FLOATS_PER_RECORD: usize = (CHANNELS * GRID_SIZE * GRID_SIZE) + META_FEATS + 1;
const BYTES_PER_RECORD: usize = FLOATS_PER_RECORD * 4;

// --- DATASET ---
struct MmapDataset {
    mmap: Arc<Mmap>,
    count: usize,
    start_index: usize,
    len: usize,
}

impl MmapDataset {
    pub fn new(path: &str) -> (Self, Self) {
        let file = File::open(path).expect("Failed to open training data file");
        let mmap = unsafe { Mmap::map(&file).expect("Failed to map file") };
        let mmap = Arc::new(mmap);

        // Read header (first 8 bytes = u64 count)
        let count_u64: u64 = pod_read_unaligned(&mmap[0..8]);
        let count = count_u64 as usize;

        // 95/5 Split
        let train_len = (count as f32 * 0.95) as usize;
        let valid_len = count - train_len;

        let train = Self {
            mmap: mmap.clone(),
            count,
            start_index: 0,
            len: train_len,
        };
        let valid = Self {
            mmap: mmap,
            count,
            start_index: train_len,
            len: valid_len,
        };

        println!(
            "Dataset Loaded. Total: {}, Train: {}, Valid: {}",
            count, train_len, valid_len
        );
        (train, valid)
    }
}

impl Dataset<Vec<f32>> for MmapDataset {
    fn get(&self, index: usize) -> Option<Vec<f32>> {
        if index >= self.len {
            return None;
        }

        let global_idx = self.start_index + index;
        // Skip 8 byte header
        let byte_offset = 8 + (global_idx * BYTES_PER_RECORD);
        let byte_end = byte_offset + BYTES_PER_RECORD;

        let bytes = &self.mmap[byte_offset..byte_end];
        let floats: &[f32] = cast_slice(bytes);

        Some(floats.to_vec())
    }

    fn len(&self) -> usize {
        self.len
    }
}

// --- BATCHING ---

#[derive(Clone, Debug)]
struct BattlesnakeBatch<B: Backend> {
    board: Tensor<B, 4>,        // [Batch, 8, 11, 11]
    metadata: Tensor<B, 2>,     // [Batch, 3]
    targets: Tensor<B, 1, Int>, // [Batch]
}

#[derive(Clone)]
struct CnnBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<B, Vec<f32>, BattlesnakeBatch<B>> for CnnBatcher<B> {
    fn batch(&self, items: Vec<Vec<f32>>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();
        let board_size = CHANNELS * GRID_SIZE * GRID_SIZE; // 968

        let mut board_vec = Vec::with_capacity(batch_size * board_size);
        let mut meta_vec = Vec::with_capacity(batch_size * META_FEATS);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            // [0 .. 968] -> Board (channel-first: [8, 11, 11])
            board_vec.extend_from_slice(&item[0..board_size]);

            // [968 .. 971] -> Metadata (3 floats)
            meta_vec.extend_from_slice(&item[board_size..board_size + META_FEATS]);

            // [971] -> Label
            // Labels: 0=up, 1=right, 2=down, 3=left (clockwise from top)
            targets_vec.push(item[board_size + META_FEATS] as i32);
        }

        // Create tensors
        let board = Tensor::from_floats(
            TensorData::new(board_vec, [batch_size, CHANNELS, GRID_SIZE, GRID_SIZE]),
            device,
        );
        let metadata =
            Tensor::from_floats(TensorData::new(meta_vec, [batch_size, META_FEATS]), device);
        let targets = Tensor::from_ints(TensorData::new(targets_vec, [batch_size]), device);

        BattlesnakeBatch {
            board,
            metadata,
            targets,
        }
    }
}

// --- TRAINING STEPS ---

impl<B: AutodiffBackend> TrainStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for BattleCnn<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.board, batch.metadata);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            ClassificationOutput {
                loss,
                output: logits,
                targets: batch.targets,
            },
        )
    }
}

impl<B: Backend> ValidStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for BattleCnn<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.board, batch.metadata);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        ClassificationOutput {
            loss,
            output: logits,
            targets: batch.targets,
        }
    }
}

// --- MAIN ---

type MyBackend = Cuda;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[tokio::main]
async fn main() {
    let device = CudaDevice::default();

    // Hyperparameters
    let batch_size = 1024;
    let learning_rate = 1e-3;
    let num_epochs = 100;

    // Model config (uses defaults, but being explicit)
    let config = BattleCnnConfig {
        in_channels: 8,
        meta_features: 3,
        num_classes: 4,
        grid_size: 11,
    };

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let model: BattleCnn<MyAutodiffBackend> = BattleCnn::new(&config, &device);

    println!("Model Architecture: 5-layer CNN with full receptive field");
    println!("Num params in model: {}", model.num_params());

    // Optimizer - AdamW with modest weight decay
    let optimizer = AdamWConfig::new().with_weight_decay(1e-4).init();

    // Load dataset
    let (dataset_train, dataset_valid) = MmapDataset::new("../preprocess/train_data_cnn.bin");

    let batcher_train = CnnBatcher::<MyAutodiffBackend> {
        device: device.clone(),
    };
    let batcher_valid = CnnBatcher::<MyBackend> {
        device: device.clone(),
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .shuffle(123)
        .num_workers(2)
        .build(dataset_valid);

    let artifact_dir = "/tmp/battlesnake-cnn";

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_checkpointing_strategy(MetricCheckpointingStrategy::new(
            &AccuracyMetric::<MyBackend>::new(),
            Aggregate::Mean,
            Direction::Highest,
            Split::Valid,
        ))
        .with_file_checkpointer(recorder.clone())
        .num_epochs(num_epochs)
        .build(model, optimizer, learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .model
        .save_file("battle_cnn_trained", &recorder)
        .expect("Failed to save model");

    println!("Training complete.");
}
