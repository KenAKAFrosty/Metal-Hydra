use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

// --- DATA STRUCTURES (Must match DB JSON) ---

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

// --- CONSTANTS ---
const GRID_SIZE: usize = 11;
const CHANNELS: usize = 8;
const META_FEATS: usize = 3;

// Channel indices
const CH_MY_HEAD: usize = 0;
const CH_ENEMY_HEADS: usize = 1;
const CH_MY_BODY: usize = 2;
const CH_ENEMY_BODIES: usize = 3;
const CH_MY_TAIL: usize = 4;
const CH_ENEMY_TAILS: usize = 5;
const CH_FOOD: usize = 6;
const CH_ENEMY_HEALTH: usize = 7;

// Board size in floats
const BOARD_SIZE: usize = CHANNELS * GRID_SIZE * GRID_SIZE; // 968

// Total floats per example: board + meta + label = 968 + 3 + 1 = 972
const FLOATS_PER_RECORD: usize = BOARD_SIZE + META_FEATS + 1;

// --- AUGMENTATION ---
// Labels: 0=up, 1=right, 2=down, 3=left (clockwise from top)
//
// The dihedral group D4 has 8 elements (4 rotations × 2 flip states)
// We apply all 8 to each training example.

/// Transform types for D4 symmetry group
#[derive(Clone, Copy)]
enum Transform {
    Identity,
    Rotate90,  // 90° clockwise
    Rotate180, // 180°
    Rotate270, // 270° clockwise (= 90° counter-clockwise)
    FlipH,     // Flip horizontal (left↔right)
    FlipV,     // Flip vertical (up↔down)
    FlipDiag,  // Flip along main diagonal (transpose)
    FlipAnti,  // Flip along anti-diagonal
}

impl Transform {
    /// All 8 transformations in D4
    fn all() -> [Transform; 8] {
        [
            Transform::Identity,
            Transform::Rotate90,
            Transform::Rotate180,
            Transform::Rotate270,
            Transform::FlipH,
            Transform::FlipV,
            Transform::FlipDiag,
            Transform::FlipAnti,
        ]
    }

    /// Transform a label (direction) according to this transformation
    /// Labels: 0=up, 1=right, 2=down, 3=left
    fn transform_label(&self, label: usize) -> usize {
        match self {
            Transform::Identity => label,
            // Rotations cycle the directions
            Transform::Rotate90 => [1, 2, 3, 0][label], // up→right, right→down, etc.
            Transform::Rotate180 => [2, 3, 0, 1][label], // up→down, right→left, etc.
            Transform::Rotate270 => [3, 0, 1, 2][label], // up→left, right→up, etc.
            // Flips swap specific pairs
            Transform::FlipH => [0, 3, 2, 1][label], // right↔left
            Transform::FlipV => [2, 1, 0, 3][label], // up↔down
            Transform::FlipDiag => [1, 0, 3, 2][label], // up↔right, down↔left
            Transform::FlipAnti => [3, 2, 1, 0][label], // up↔left, down↔right
        }
    }

    /// Transform (x, y) coordinates according to this transformation
    /// Assumes grid is GRID_SIZE × GRID_SIZE
    fn transform_coords(&self, x: i32, y: i32) -> (i32, i32) {
        let max = GRID_SIZE as i32 - 1;
        match self {
            Transform::Identity => (x, y),
            Transform::Rotate90 => (max - y, x),
            Transform::Rotate180 => (max - x, max - y),
            Transform::Rotate270 => (y, max - x),
            Transform::FlipH => (max - x, y),
            Transform::FlipV => (x, max - y),
            Transform::FlipDiag => (y, x),
            Transform::FlipAnti => (max - y, max - x),
        }
    }
}

fn main() {
    let db_path = "../battlesnake_data.db"; // Adjust if needed
    let out_path = "train_data_cnn_augmented.bin";

    println!("Loading data from {}...", db_path);
    let conn = Connection::open(db_path).expect("Could not open DB");

    let mut stmt = conn
        .prepare("SELECT data_json FROM training_examples")
        .expect("Table not found");

    let rows = stmt
        .query_map([], |row| {
            let json: String = row.get(0)?;
            Ok(serde_json::from_str::<TrainingExample>(&json).unwrap())
        })
        .expect("Query failed");

    let f = File::create(out_path).expect("Unable to create file");
    let mut writer = BufWriter::new(f);

    // 1. Write Header Placeholder (8 bytes for u64 count)
    writer.write_all(&0u64.to_le_bytes()).unwrap();

    let mut count: u64 = 0;
    let mut skipped: u64 = 0;

    // Re-use these buffers to avoid allocation
    let mut buffer = vec![0.0f32; FLOATS_PER_RECORD];

    println!("Processing and writing records (with 8x augmentation)...");
    for (i, item) in rows.enumerate() {
        if i % 10_000 == 0 {
            print!(
                "\rProcessed {} raw examples, wrote {} augmented...",
                i, count
            );
            std::io::stdout().flush().unwrap();
        }
        let item = item.unwrap();

        let w = item.game.width as usize;
        let h = item.game.height as usize;
        let area = (w * h) as f32;

        // Find the winning snake (our perspective)
        let target_id = &item.winning_snake_id;
        let my_snake = item
            .snakes
            .iter()
            .find(|s| s.id == *target_id && s.death.is_none());

        let my_snake = match my_snake {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };

        let my_len = my_snake.body.len();
        let my_health = my_snake.health as f32 / 100.0;

        // Metadata (same for all transforms)
        let food_chance = item
            .game
            .ruleset
            .food_spawn_chance
            .parse::<f32>()
            .unwrap_or(15.0);
        let min_food = item.game.ruleset.minimum_food.parse::<f32>().unwrap_or(1.0);

        let meta = [food_chance / 100.0, min_food / area, my_health];

        // Original label
        let orig_label = item.label;

        // Apply all 8 transformations
        for transform in Transform::all() {
            buffer.fill(0.0);

            // Encode board with transformation
            encode_board_transformed(&item, my_snake, my_len, &mut buffer, transform);

            // Write metadata
            let meta_start = BOARD_SIZE;
            buffer[meta_start + 0] = meta[0];
            buffer[meta_start + 1] = meta[1];
            buffer[meta_start + 2] = meta[2];

            // Write transformed label
            let transformed_label = transform.transform_label(orig_label);
            buffer[meta_start + 3] = transformed_label as f32;

            // Write to file
            let bytes = bytemuck::cast_slice(&buffer);
            writer.write_all(bytes).unwrap();

            count += 1;
        }
    }

    // Update the header count
    writer.flush().unwrap();
    let mut f = writer.into_inner().unwrap();
    f.seek(std::io::SeekFrom::Start(0)).unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();

    println!(
        "\nDone! Wrote {} records to {}. Skipped {} raw examples.",
        count, out_path, skipped
    );
    println!(
        "File size: {:.2} GB",
        (count as usize * FLOATS_PER_RECORD * 4) as f64 / 1e9
    );
    println!(
        "Original examples: ~{}, with 8x augmentation: {}",
        count / 8,
        count
    );
}

fn encode_board_transformed(
    item: &TrainingExample,
    my_snake: &Snake,
    my_len: usize,
    buffer: &mut [f32],
    transform: Transform,
) {
    let target_id = &my_snake.id;

    // Process all snakes
    for snake in &item.snakes {
        if snake.death.is_some() {
            continue;
        }

        let len = snake.body.len();
        let is_me = snake.id == *target_id;
        let health_norm = snake.health as f32 / 100.0;

        // === HEAD ===
        let head = &snake.body[0];
        let (tx, ty) = transform.transform_coords(head.x, head.y);
        if in_bounds(tx, ty) {
            if is_me {
                set_channel(buffer, CH_MY_HEAD, tx, ty, 1.0);
            } else {
                let relative_size = len as f32 / (len + my_len) as f32;
                set_channel(buffer, CH_ENEMY_HEADS, tx, ty, relative_size);
                set_channel(buffer, CH_ENEMY_HEALTH, tx, ty, health_norm);
            }
        }

        // === BODY (excluding head and tail) ===
        if len > 2 {
            for (i, part) in snake.body.iter().enumerate().skip(1).take(len - 2) {
                let (tx, ty) = transform.transform_coords(part.x, part.y);
                if !in_bounds(tx, ty) {
                    continue;
                }

                let gradient = (len - i) as f32 / len as f32;

                if is_me {
                    set_channel(buffer, CH_MY_BODY, tx, ty, gradient);
                } else {
                    set_channel(buffer, CH_ENEMY_BODIES, tx, ty, gradient);
                    set_channel(buffer, CH_ENEMY_HEALTH, tx, ty, health_norm);
                }
            }
        }

        // === TAIL ===
        if len > 1 {
            if let Some(tail) = snake.body.last() {
                let (tx, ty) = transform.transform_coords(tail.x, tail.y);
                if in_bounds(tx, ty) {
                    if is_me {
                        set_channel(buffer, CH_MY_TAIL, tx, ty, 1.0);
                    } else {
                        set_channel(buffer, CH_ENEMY_TAILS, tx, ty, 1.0);
                        set_channel(buffer, CH_ENEMY_HEALTH, tx, ty, health_norm);
                    }
                }
            }
        }
    }

    // === FOOD ===
    for food in &item.food {
        let (tx, ty) = transform.transform_coords(food.x, food.y);
        if in_bounds(tx, ty) {
            set_channel(buffer, CH_FOOD, tx, ty, 1.0);
        }
    }
}

#[inline]
fn in_bounds(x: i32, y: i32) -> bool {
    x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32
}

#[inline]
fn set_channel(buffer: &mut [f32], channel: usize, x: i32, y: i32, value: f32) {
    let idx = (channel * GRID_SIZE * GRID_SIZE) + (y as usize * GRID_SIZE) + x as usize;
    buffer[idx] = value;
}
