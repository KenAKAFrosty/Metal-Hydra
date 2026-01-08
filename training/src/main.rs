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
use burn::train::metric::LossMetric; // FIXED: Removed NumericMetric
use burn::train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep};
use bytemuck::{cast_slice, pod_read_unaligned};
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;

// Update to point to your new module
use burn_ai_model::transformer_winprob::{BattleModel, BattleModelConfig};

// --- CONSTANTS ---
const GRID_SIZE: usize = 11;
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE;
const TILE_FEATS: usize = 27;
const META_FEATS: usize = 2;
const FLOATS_PER_RECORD: usize = (SEQ_LEN * TILE_FEATS) + META_FEATS + 4;
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
        let file = File::open(path).expect("Failed to open train_data_value.bin");
        let mmap = unsafe { Mmap::map(&file).expect("Failed to map file") };
        let mmap = Arc::new(mmap);

        let count_u64: u64 = pod_read_unaligned(&mmap[0..8]);
        let count = count_u64 as usize;

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
    tiles: Tensor<B, 3>,
    metadata: Tensor<B, 2>,
    targets: Tensor<B, 2>, // [Batch, 4]
}

#[derive(Clone)]
struct BinaryBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<B, Vec<f32>, BattlesnakeBatch<B>> for BinaryBatcher<B> {
    fn batch(&self, items: Vec<Vec<f32>>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();

        let mut tiles_vec = Vec::with_capacity(batch_size * SEQ_LEN * TILE_FEATS);
        let mut meta_vec = Vec::with_capacity(batch_size * META_FEATS);
        // Capacity = batch_size * 4 outputs
        let mut targets_vec = Vec::with_capacity(batch_size * 4);

        let split_idx = SEQ_LEN * TILE_FEATS;

        for item in items {
            // Tiles
            tiles_vec.extend_from_slice(&item[0..split_idx]);

            // Metadata (2 floats)
            meta_vec.extend_from_slice(&item[split_idx..split_idx + 2]);

            // Targets (4 floats)
            // split_idx + 2  -> Up
            // split_idx + 3  -> Down
            // split_idx + 4  -> Right
            // split_idx + 5  -> Left
            targets_vec.extend_from_slice(&item[split_idx + 2..split_idx + 6]);
        }

        let tiles = Tensor::from_floats(
            TensorData::new(tiles_vec, [batch_size, SEQ_LEN, TILE_FEATS]),
            device,
        );
        let metadata =
            Tensor::from_floats(TensorData::new(meta_vec, [batch_size, META_FEATS]), device);

        // Shape: [Batch, 4]
        let targets = Tensor::from_floats(TensorData::new(targets_vec, [batch_size, 4]), device);

        BattlesnakeBatch {
            tiles,
            metadata,
            targets,
        }
    }
}

// --- TRAINING STEPS ---
impl<B: AutodiffBackend> TrainStep<BattlesnakeBatch<B>, RegressionOutput<B>> for BattleModel<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        // 1. Forward Pass
        // Output Shape: [Batch, 4]
        let preds = self.forward(batch.tiles, batch.metadata);

        // 2. Loss Calculation
        // Compare the full vectors.
        // Taken action matches Target (e.g. 0.8).
        // Untaken actions match 0.0.
        let loss = burn::nn::loss::MseLoss::new().forward(
            preds.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        // 3. Backward Pass
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            RegressionOutput {
                loss,
                output: preds,
                targets: batch.targets,
            },
        )
    }
}

// Don't forget to update ValidStep similarly!
impl<B: Backend> ValidStep<BattlesnakeBatch<B>, RegressionOutput<B>> for BattleModel<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> RegressionOutput<B> {
        let preds = self.forward(batch.tiles, batch.metadata);

        let loss = burn::nn::loss::MseLoss::new().forward(
            preds.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        RegressionOutput {
            loss,
            output: preds,
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

    let batch_size = 528;
    let learning_rate = 3e-4;
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
        dropout: 0.2,
        head_hidden_size: 128,
        num_queries: 8,
    };

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model: BattleModel<MyAutodiffBackend> = BattleModel::new(&config, &device);

    println!("Num params in model: {}", model.num_params());

    let optimizer = AdamWConfig::new()
        .with_weight_decay(1e-4)
        // .with_cautious_weight_decay(true)
        .init();

    let (dataset_train, dataset_valid) = MmapDataset::new("../preprocess/train_data_value.bin");

    let batcher = BinaryBatcher::<MyAutodiffBackend> {
        device: device.clone(),
    };
    let batcher_valid = BinaryBatcher::<MyBackend> {
        device: device.clone(),
    };

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(2)
        .build(dataset_valid);

    let artifact_dir = "/tmp/battlesnake-transformer-value";

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_checkpointing_strategy(MetricCheckpointingStrategy::new(
            &LossMetric::<MyBackend>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
        ))
        .with_file_checkpointer(recorder.clone())
        .num_epochs(num_epochs)
        .build(model, optimizer, learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .model
        .save_file("transformer_value", &recorder)
        .expect("Failed to save model");

    println!("Training complete.");
}
