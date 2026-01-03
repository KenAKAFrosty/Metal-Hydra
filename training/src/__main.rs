#![recursion_limit = "256"]

use burn::backend::cuda::{Cuda, CudaDevice};
// use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use burn_ai_model::transformer::{BattleModel, BattleModelConfig};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct DeathInfo {
    turn: u32,
    cause: String,
    eliminated_by: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Snake {
    #[serde(rename = "ID")]
    id: String,
    body: Vec<Position>,
    health: i32,
    is_bot: bool,
    death: Option<DeathInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct Ruleset {
    food_spawn_chance: String,
    minimum_food: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Game {
    width: u32,
    height: u32,
    ruleset: Ruleset,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TrainingExample {
    #[serde(rename = "Game")]
    game: Game,
    #[serde(rename = "Snakes")]
    snakes: Vec<Snake>,
    #[serde(rename = "Food")]
    food: Vec<Position>,
    #[serde(rename = "Hazards")]
    hazards: Vec<Position>,
    #[serde(rename = "Turn")]
    turn: u32,
    label: usize,
    winning_snake_id: String,
}

#[derive(Clone)]
struct BattlesnakeDataset {
    items: Vec<TrainingExample>,
}

impl BattlesnakeDataset {
    pub fn new(items: Vec<TrainingExample>) -> Self {
        Self { items }
    }
}

impl Dataset<TrainingExample> for BattlesnakeDataset {
    fn get(&self, index: usize) -> Option<TrainingExample> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

fn load_data(db_path: &str) -> Vec<TrainingExample> {
    println!("Loading data from {}...", db_path);
    let conn = Connection::open(db_path).expect("Could not open DB");
    let mut stmt = conn
        .prepare("SELECT data_json FROM training_examples")
        .expect("Table not found");
    let items_iter = stmt
        .query_map([], |row| {
            let json: String = row.get(0)?;
            Ok(serde_json::from_str::<TrainingExample>(&json).unwrap())
        })
        .expect("Query failed");

    let mut items = Vec::new();
    for item in items_iter {
        if let Ok(ex) = item {
            items.push(ex);
        }
    }
    println!("Total examples loaded: {}", items.len());
    items
}

type MyBackend = Cuda;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Clone)]
struct BattlesnakeBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct BattlesnakeBatch<B: Backend> {
    // [Batch, 121, 26] - sequence of tiles
    tiles: Tensor<B, 3>,
    // [Batch, 4] - global metadata
    metadata: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

const GRID_SIZE: usize = 11;
const SEQUENCE_LENGTH: usize = GRID_SIZE * GRID_SIZE;
const TILE_FEATURE_COUNT: usize = 26;
const META_FEATURE_COUNT: usize = 2;

// helper to set features in the flat vector
// grid: flat vector of size 121 * 26
// idx: 0..120 (tile index)
// feat: 0..25 (feature index)
// val: value to set
// fn set_feature(grid: &mut [f32], x: i32, y: i32, feature: usize, value: f32) {
//     let tile_idx = (y as usize * GRID_SIZE) + x as usize;
//     let vec_idx = (tile_idx * TILE_FEATURE_COUNT) + feature;
//     grid[vec_idx] = value;
//     // if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
//     // removed this to avoid excessive bounds checking. we just presume these are valid games
//     // }
// }

impl<B: Backend> Batcher<B, TrainingExample, BattlesnakeBatch<B>> for BattlesnakeBatcher<B> {
    fn batch(&self, items: Vec<TrainingExample>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();

        // 1. Calculate total size
        let total_features = batch_size * SEQUENCE_LENGTH * TILE_FEATURE_COUNT;

        // 2. Pre-allocate AND initialize with zeros immediately
        // This is one big allocation instead of 256 small ones.
        let mut tiles_data = vec![0.0f32; total_features];

        let mut meta_data = Vec::with_capacity(batch_size * META_FEATURE_COUNT);
        let mut targets_data = Vec::with_capacity(batch_size);

        // We use a stride to jump through the flat tiles_data vector
        let game_stride = SEQUENCE_LENGTH * TILE_FEATURE_COUNT;

        for (batch_idx, item) in items.into_iter().enumerate() {
            let w = item.game.width as usize; // Usually 11
            let h = item.game.height as usize;
            let area = (w * h) as f32;

            // Calculate the start index for this specific game in the giant vector
            let base_offset = batch_idx * game_stride;

            // --- INLINED HELPER ---
            // We use a closure that captures `base_offset` and `tiles_data` slice
            // to write directly into the main memory.
            let mut write_feature = |x: i32, y: i32, feature_offset: usize, value: f32| {
                // trusting data is valid (0..10) for speed, or keep the check if unsure
                let tile_idx = (y as usize * GRID_SIZE) + x as usize;
                // distinct feature index
                let local_idx = (tile_idx * TILE_FEATURE_COUNT) + feature_offset;
                // Write directly to the pre-zeroed buffer
                tiles_data[base_offset + local_idx] = value;
            };

            let target_id = &item.winning_snake_id;
            let mut my_snake: Option<Snake> = None;
            let mut enemies: Vec<Snake> = Vec::with_capacity(3);

            // Move ownership instead of referencing to avoid lifetime issues in iterator
            for snake in item.snakes {
                if snake.death.is_some() {
                    continue;
                }
                if snake.id == *target_id {
                    my_snake = Some(snake);
                } else {
                    enemies.push(snake);
                }
            }

            // --- PROCESS SNAKES ---
            // Define logic once
            let mut process_body = |snake: &Snake, offset: usize| {
                let len = snake.body.len();
                let norm_health = snake.health as f32 / 100.0;
                let norm_len = len as f32 / area;

                for (i, part) in snake.body.iter().enumerate() {
                    // 1. ANCHORS
                    if i == 0 {
                        write_feature(part.x, part.y, offset + 0, 1.0);
                    } else if i == 1 {
                        write_feature(part.x, part.y, offset + 1, 1.0);
                    } else if i == len - 1 {
                        write_feature(part.x, part.y, offset + 2, 1.0);
                    } else {
                        let gradient = (len - i) as f32 / len as f32;
                        write_feature(part.x, part.y, offset + 3, gradient);
                    }

                    // 2. STATS
                    write_feature(part.x, part.y, offset + 4, norm_health);
                    write_feature(part.x, part.y, offset + 5, norm_len);
                }
            };

            if let Some(me) = my_snake {
                process_body(&me, 0);
            }

            enemies.sort_by(|a, b| a.id.cmp(&b.id));
            for (i, enemy) in enemies.iter().take(3).enumerate() {
                process_body(enemy, 6 + (i * 6));
            }

            // --- GLOBAL FEATURES ---
            for food in item.food {
                write_feature(food.x, food.y, 24, 1.0);
            }
            for hazard in item.hazards {
                write_feature(hazard.x, hazard.y, 25, 1.0);
            }

            // --- METADATA ---
            let food_spawn_chance: f32 =
                item.game.ruleset.food_spawn_chance.parse().unwrap_or(15.0);
            let min_food: f32 = item.game.ruleset.minimum_food.parse().unwrap_or(1.0);

            meta_data.push(food_spawn_chance / 100.0);
            meta_data.push(min_food / area);
            targets_data.push(item.label as i32);
        }

        let tiles_shape = [batch_size, SEQUENCE_LENGTH, TILE_FEATURE_COUNT];
        let tiles = Tensor::from_data(TensorData::new(tiles_data, tiles_shape), &self.device);

        let meta_shape = [batch_size, META_FEATURE_COUNT];
        let metadata = Tensor::from_data(TensorData::new(meta_data, meta_shape), &self.device);

        let targets = Tensor::from_data(TensorData::new(targets_data, [batch_size]), &self.device);

        BattlesnakeBatch {
            tiles,
            metadata,
            targets,
        }
    }
}

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

#[tokio::main]
async fn main() {
    let device = CudaDevice::default();

    let batch_size = 64;
    let learning_rate = 3e-5;
    let num_epochs = 100;

    let config = BattleModelConfig {
        d_model: 512, // Embedding size
        d_ff: 2048,   // Feed forward inner dimension
        n_heads: 8,   // Attention heads. rule of thumb is d_model / 64 = n_heads
        n_layers: 10,
        num_classes: 4,
        tile_features: 26, // Match Batcher
        meta_features: 2,  // Match Batcher
        grid_size: 11,
    };

    let model: BattleModel<MyAutodiffBackend> = BattleModel::new(&config, &device);

    // // 2. Load the existing weights
    // println!("Loading pre-trained weights...");
    // let record = CompactRecorder::new()
    //     .load("transformer_v1_large_sft1".into(), &device) // Make sure path matches your saved file
    //     .expect("Failed to load pre-trained model weights");

    // // 3. Apply the weights to the model
    // let model = model.load_record(record);

    let optimizer = AdamWConfig::new().with_cautious_weight_decay(true).init();

    let mut all_data = load_data("../battlesnake_data.db");
    if all_data.len() <= batch_size {
        panic!("Not enough data!");
    }

    let valid_data = all_data.split_off(all_data.len() - batch_size);
    let train_data = all_data;

    let dataset_train = BattlesnakeDataset::new(train_data);
    let dataset_valid = BattlesnakeDataset::new(valid_data);

    let batcher_train = BattlesnakeBatcher::<MyAutodiffBackend> {
        device: device.clone(),
    };
    let batcher_valid = BattlesnakeBatcher::<MyBackend> {
        device: device.clone(),
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(dataset_valid);

    let artifact_dir = "/tmp/battlesnake-transformer";

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .build(model, optimizer, learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .model
        .save_file("transformer_v3", &CompactRecorder::new())
        .expect("Failed to save model");

    println!("Transformer training complete!");
}
