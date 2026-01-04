#![recursion_limit = "256"]

use burn::backend::{cuda::Cuda, cuda::CudaDevice, Autodiff}; // USING CUDA
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::checkpoint::{FileCheckpointer, MetricCheckpointingStrategy};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, TrainOutput, TrainStep,
    ValidStep,
};
use bytemuck::{cast_slice, pod_read_unaligned};
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;

// Import your model
use burn_ai_model::transformer::{BattleModel, BattleModelConfig};

// --- DATASET & BATCHER LOGIC ---

const GRID_SIZE: usize = 11;
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE;
const TILE_FEATS: usize = 27;
const META_FEATS: usize = 2;
// (121 * 26) + 2 + 1 = 3149
const FLOATS_PER_RECORD: usize = (SEQ_LEN * TILE_FEATS) + META_FEATS + 1;
const BYTES_PER_RECORD: usize = FLOATS_PER_RECORD * 4;

struct MmapDataset {
    mmap: Arc<Mmap>,
    count: usize,
    start_index: usize,
    len: usize,
}

impl MmapDataset {
    pub fn new(path: &str) -> (Self, Self) {
        let file = File::open(path).expect("Failed to open train_data.bin");
        // SAFETY: We assume the file is not modified by another process while running
        let mmap = unsafe { Mmap::map(&file).expect("Failed to map file") };
        let mmap = Arc::new(mmap);

        // Read header (first 8 bytes = u64 count)
        let count_u64: u64 = pod_read_unaligned(&mmap[0..8]);
        let count = count_u64 as usize;

        // 95/5 Split (Since you have 1.8M examples, 5% is plenty for validation)
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

// Return Vec<f32> to the Dataloader
impl Dataset<Vec<f32>> for MmapDataset {
    fn get(&self, index: usize) -> Option<Vec<f32>> {
        if index >= self.len {
            return None;
        }

        let global_idx = self.start_index + index;
        // Skip 8 byte header
        let byte_offset = 8 + (global_idx * BYTES_PER_RECORD);
        let byte_end = byte_offset + BYTES_PER_RECORD;

        // Zero-copy read from OS cache
        let bytes = &self.mmap[byte_offset..byte_end];
        let floats: &[f32] = cast_slice(bytes);

        // We clone into a Vec here. This is a very fast memcpy.
        Some(floats.to_vec())
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
struct BattlesnakeBatch<B: Backend> {
    tiles: Tensor<B, 3>,    // [Batch, 121, 26]
    metadata: Tensor<B, 2>, // [Batch, 2]
    targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
struct BinaryBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<B, Vec<f32>, BattlesnakeBatch<B>> for BinaryBatcher<B> {
    fn batch(&self, items: Vec<Vec<f32>>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();

        // 1. Separate vectors to ensure memory layout is perfect for conversion
        let mut tiles_vec = Vec::with_capacity(batch_size * SEQ_LEN * TILE_FEATS);
        let mut meta_vec = Vec::with_capacity(batch_size * META_FEATS);
        let mut targets_vec = Vec::with_capacity(batch_size);

        let split_idx = SEQ_LEN * TILE_FEATS;

        for item in items {
            // [0 .. 3146] -> Tiles
            tiles_vec.extend_from_slice(&item[0..split_idx]);
            // [3146 .. 3148] -> Meta
            meta_vec.extend_from_slice(&item[split_idx..split_idx + 2]);
            // [3148] -> Label
            targets_vec.push(item[split_idx + 2] as i32);
        }

        // 2. To Device
        let tiles = Tensor::from_floats(
            TensorData::new(tiles_vec, [batch_size, SEQ_LEN, TILE_FEATS]),
            device,
        );
        let metadata =
            Tensor::from_floats(TensorData::new(meta_vec, [batch_size, META_FEATS]), device);
        let targets = Tensor::from_ints(TensorData::new(targets_vec, [batch_size]), device);

        BattlesnakeBatch {
            tiles,
            metadata,
            targets,
        }
    }
}

// --- TRAINING STEPS ---

impl<B: AutodiffBackend> TrainStep<BattlesnakeBatch<B>, ClassificationOutput<B>>
    for BattleModel<B>
{
    fn step(&self, batch: BattlesnakeBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.tiles, batch.metadata);

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

impl<B: Backend> ValidStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for BattleModel<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.tiles, batch.metadata);

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

    let batch_size = 1024;
    let learning_rate = 6e-4;
    let num_epochs = 100;

    let config = BattleModelConfig {
        d_model: 64,
        d_ff: 256,
        n_heads: 2,
        n_layers: 4,
        num_classes: 4,
        tile_features: 27,
        meta_features: 2,
        grid_size: 11,
        dropout: 0.1,
        head_hidden_size: 128,
        num_queries: 8,
    };

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let model: BattleModel<MyAutodiffBackend> = BattleModel::new(&config, &device);

    println!("Num params in model: {}", model.num_params());

    // let model = model
    //     .load_file("../transformer_optimized", &CompactRecorder::new(), &device)
    //     .expect("Could not load model weights! Check path and config compatibility.");

    let optimizer = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .with_cautious_weight_decay(true)
        .init();

    // --- LOAD DATASET (BINARY) ---
    // Make sure preprocess has run and generated this file!
    let (dataset_train, dataset_valid) = MmapDataset::new("../preprocess/train_data.bin");

    let batcher = BinaryBatcher::<MyAutodiffBackend> {
        device: device.clone(),
    };

    let batcher_valid = BinaryBatcher::<MyBackend> {
        device: device.clone(),
    };

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4) // Parallel disk reads
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(2)
        .build(dataset_valid);

    let artifact_dir = "/tmp/battlesnake-transformer-drone";

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
        .save_file("transformer_drone", &recorder)
        .expect("Failed to save model");

    println!("Training complete.");
}
