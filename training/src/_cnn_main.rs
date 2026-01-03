use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use burn_ai_model::cnn_v4_beef_to_the_future::ModelBeefierCnn as Model;

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

// ------------------------------------------------------------------
// 2. Dataset
// ------------------------------------------------------------------

#[derive(Clone)]
struct BattlesnakeDataset {
    items: Vec<TrainingExample>,
}

impl BattlesnakeDataset {
    // UPDATED: Now takes a Vec directly, logic moved to main or helper
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

// Helper to load from DB
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

// ------------------------------------------------------------------
// 3. Batcher
// ------------------------------------------------------------------

// Define specific backends to avoid ambiguity
type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Clone)]
struct BattlesnakeBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct BattlesnakeBatch<B: Backend> {
    board: Tensor<B, 4>,        // [Batch, 8, 11, 11]
    metadata: Tensor<B, 2>,     // [Batch, 8]
    targets: Tensor<B, 1, Int>, // [Batch]
}

const CHANNELS: usize = 8;
const HEIGHT: usize = 11;
const WIDTH: usize = 11;
const METADATA_COUNT: usize = 8;

// Helper to write to the grid (fixes Borrow Checker error)
fn write_grid(grid: &mut [f32], ch: usize, y: i32, x: i32, val: f32, accumulate: bool) {
    if x >= 0 && x < WIDTH as i32 && y >= 0 && y < HEIGHT as i32 {
        let idx = (ch * HEIGHT * WIDTH) + (y as usize * WIDTH) + x as usize;
        if accumulate {
            grid[idx] += val;
        } else {
            grid[idx] = val;
        }
    }
}

impl<B: Backend> Batcher<B, TrainingExample, BattlesnakeBatch<B>> for BattlesnakeBatcher<B> {
    fn batch(&self, items: Vec<TrainingExample>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();

        let mut board_data = Vec::with_capacity(batch_size * CHANNELS * HEIGHT * WIDTH);
        let mut meta_data = Vec::with_capacity(batch_size * METADATA_COUNT);
        let mut targets_data = Vec::with_capacity(batch_size);

        for item in items {
            let w = item.game.width as usize;
            let h = item.game.height as usize;
            let area = (w * h) as f32;
            let target_id = &item.winning_snake_id;

            // --- Preprocessing ---
            let mut grid = vec![0.0f32; CHANNELS * HEIGHT * WIDTH];

            let mut other_len_sum = 0usize;
            let mut other_count = 0usize;
            let mut longest_other = 0usize;
            let mut my_snake: Option<&Snake> = None;

            for snake in &item.snakes {
                // Filter dead snakes
                if snake.death.is_some() {
                    continue;
                }

                let len = snake.body.len();
                if len == 0 {
                    continue;
                } // Sanity check

                let norm_len = len as f32 / area;
                let norm_health = snake.health as f32 / 100.0;
                let is_target = snake.id == *target_id;

                if is_target {
                    my_snake = Some(snake);
                } else {
                    other_len_sum += len;
                    other_count += 1;
                    longest_other = longest_other.max(len);
                }

                // Head
                let head = &snake.body[0];
                let head_ch = if is_target { 0 } else { 1 };
                write_grid(&mut grid, head_ch, head.y, head.x, norm_len, false);

                // Body
                let body_ch = if is_target { 2 } else { 3 };
                for part in snake.body.iter().skip(1).take(len.saturating_sub(2)) {
                    write_grid(&mut grid, body_ch, part.y, part.x, norm_health, false);
                }

                // Tail
                if let Some(tail) = snake.body.last() {
                    let tail_ch = if is_target { 4 } else { 5 };
                    write_grid(&mut grid, tail_ch, tail.y, tail.x, norm_health, false);
                }
            }

            // Food
            for f in &item.food {
                write_grid(&mut grid, 6, f.y, f.x, 1.0, true);
            }

            // Hazards
            for hz in &item.hazards {
                write_grid(&mut grid, 7, hz.y, hz.x, 1.0, true);
            }

            // --- Metadata ---
            let fs_chance: f32 = item.game.ruleset.food_spawn_chance.parse().unwrap_or(15.0);
            let min_food: f32 = item.game.ruleset.minimum_food.parse().unwrap_or(1.0);

            let food_spawn = fs_chance / 100.0;
            let min_food_norm = min_food / area;
            let turn_norm = item.turn as f32 / 7200.0;

            let (head_x, head_y, health, target_len) = if let Some(me) = my_snake {
                if !me.body.is_empty() {
                    (
                        me.body[0].x as f32 / w as f32,
                        me.body[0].y as f32 / h as f32,
                        me.health as f32 / 100.0,
                        me.body.len(),
                    )
                } else {
                    (0.0, 0.0, 0.0, 0)
                }
            } else {
                (0.0, 0.0, 0.0, 0)
            };

            let is_longest = if target_len > longest_other { 1.0 } else { 0.0 };
            let avg_frac = if is_longest > 0.5 && other_count > 0 {
                (other_len_sum as f32 / other_count as f32) / target_len as f32
            } else {
                0.0
            };

            board_data.extend(grid);
            meta_data.extend(vec![
                food_spawn,
                min_food_norm,
                turn_norm,
                head_x,
                head_y,
                health,
                is_longest,
                avg_frac,
            ]);
            targets_data.push(item.label as i32);
        }

        let board_shape = [batch_size, CHANNELS, HEIGHT, WIDTH];
        let board_tensor =
            Tensor::from_data(TensorData::new(board_data, board_shape), &self.device);

        let meta_shape = [batch_size, METADATA_COUNT];
        let meta_tensor = Tensor::from_data(TensorData::new(meta_data, meta_shape), &self.device);

        let targets_tensor =
            Tensor::from_data(TensorData::new(targets_data, [batch_size]), &self.device);

        BattlesnakeBatch {
            board: board_tensor,
            metadata: meta_tensor,
            targets: targets_tensor,
        }
    }
}

// ------------------------------------------------------------------
// 4. Training Step
// ------------------------------------------------------------------

impl<B: AutodiffBackend> TrainStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for Model<B> {
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

impl<B: Backend> ValidStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for Model<B> {
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

// ------------------------------------------------------------------
// 5. Main
// ------------------------------------------------------------------

use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};

#[tokio::main]
async fn main() {
    let device = WgpuDevice::default();
    let batch_size = 256;
    let learning_rate = 2e-4;
    let num_epochs = 25;
    let artifact_dir = "/tmp/battlesnake-model";

    // Load Data
    // 1. Load Data
    let mut all_data = load_data("../battlesnake_data.db");

    // 2. "Hack" to skip validation time:
    // We split off just ONE batch (64 items) to be the "Validation Set".
    // This satisfies the Learner API but takes 0.001s to process.
    if all_data.len() <= batch_size {
        panic!(
            "Not enough data to run! Need at least {} items.",
            batch_size * 2
        );
    }
    let valid_data = all_data.split_off(all_data.len() - batch_size);
    let train_data = all_data; // The rest is training

    println!("Training Set: {} items", train_data.len());
    println!("Validation Set: {} items (Dummy split)", valid_data.len());

    let dataset_train = BattlesnakeDataset::new(train_data);
    let dataset_valid = BattlesnakeDataset::new(valid_data);

    // 3. Create Dataloaders
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
        // No need to shuffle validation
        .num_workers(1)
        .build(dataset_valid);

    // Initialize Model
    let model: Model<MyAutodiffBackend> = Model::new(&device);

    let optimizer = AdamWConfig::new().init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .build(model, optimizer, learning_rate);

    // Train
    let model_trained = learner.fit(
        dataloader_train,
        dataloader_valid, // FIX 3: Pass the valid dataloader here!
    );

    model_trained
        .model
        .save_file("v4beefbeef_325kX", &CompactRecorder::new())
        .expect("Failed to save model");

    println!("Training complete!");
}
