use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

// --- DATA STRUCTURES (Unchanged) ---
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

// --- CONSTANTS (Unchanged) ---
const GRID_SIZE: usize = 11;
const CHANNELS: usize = 8;
const META_FEATS: usize = 3;

// Channel definitions (Unchanged)
const CH_MY_HEAD: usize = 0;
const CH_ENEMY_HEADS: usize = 1;
const CH_MY_BODY: usize = 2;
const CH_ENEMY_BODIES: usize = 3;
const CH_MY_TAIL: usize = 4;
const CH_ENEMY_TAILS: usize = 5;
const CH_FOOD: usize = 6;
const CH_ENEMY_HEALTH: usize = 7;

const BOARD_SIZE: usize = CHANNELS * GRID_SIZE * GRID_SIZE;
const FLOATS_PER_RECORD: usize = BOARD_SIZE + META_FEATS + 1;

fn main() {
    let db_path = "../battlesnake_data.db";
    // Updated filename to reflect augmentation
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

    writer.write_all(&0u64.to_le_bytes()).unwrap();

    let mut count: u64 = 0;
    let mut skipped_winner_dead: u64 = 0;

    // Reuse buffer
    let mut buffer = vec![0.0f32; FLOATS_PER_RECORD];

    println!("Processing records with D4 Symmetry Augmentation (8x expansion)...");

    for (i, item) in rows.enumerate() {
        if i % 1000 == 0 {
            print!(
                "\rProcessed {} raw games (approx {} samples written)...",
                i, count
            );
            std::io::stdout().flush().unwrap();
        }
        let item = item.unwrap();

        // 1. Basic validation (width/height check strictly for augmentation logic)
        let w = item.game.width as usize;
        let h = item.game.height as usize;

        // This augmentation logic assumes a square grid usually,
        // but works for rectangles if you handle W/H swapping on 90 deg rotations.
        // For standard battlesnake, it's 11x11.
        if w != GRID_SIZE || h != GRID_SIZE {
            continue;
        }

        let area = (w * h) as f32;

        let target_id = &item.winning_snake_id;
        let my_snake = item
            .snakes
            .iter()
            .find(|s| s.id == *target_id && s.death.is_none());

        let my_snake = match my_snake {
            Some(s) => s,
            None => {
                skipped_winner_dead += 1;
                continue;
            }
        };

        let my_len = my_snake.body.len();
        let my_health = my_snake.health as f32 / 100.0;

        // Metadata (Ruleset)
        let food_chance = item
            .game
            .ruleset
            .food_spawn_chance
            .parse::<f32>()
            .unwrap_or(15.0);
        let min_food = item.game.ruleset.minimum_food.parse::<f32>().unwrap_or(1.0);

        // --- AUGMENTATION LOOP ---
        // We generate 8 variations for every single valid game state
        for symmetry_mode in 0..8 {
            // 1. Clear buffer
            buffer.fill(0.0);

            // 2. Encode Board with coordinate transform
            encode_board(&item, my_snake, my_len, &mut buffer, symmetry_mode);

            // 3. Write Metadata (invariant to rotation)
            buffer[BOARD_SIZE + 0] = food_chance / 100.0;
            buffer[BOARD_SIZE + 1] = min_food / area;
            buffer[BOARD_SIZE + 2] = my_health;

            // 4. Write Label (must be transformed!)
            let augmented_label = transform_label(item.label, symmetry_mode);
            buffer[BOARD_SIZE + 3] = augmented_label as f32;

            // 5. Write to disk
            let bytes = bytemuck::cast_slice(&buffer);
            writer.write_all(bytes).unwrap();
            count += 1;
        }
    }

    writer.flush().unwrap();
    let mut f = writer.into_inner().unwrap();
    f.seek(std::io::SeekFrom::Start(0)).unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();

    println!("\nDone! Wrote {} records to {}.", count, out_path);
    println!("Skipped: {} (winner dead)", skipped_winner_dead);
}

// --- AUGMENTATION HELPERS ---

// Transforms a point (x,y) based on the symmetry mode (0-7)
// Grid is assumed 11x11 (0..10).
#[inline]
fn transform_pos(x: i32, y: i32, mode: u8) -> (i32, i32) {
    let max = (GRID_SIZE - 1) as i32;

    // First apply the flip (if mode >= 4)
    let (tx, ty) = if mode >= 4 {
        (max - x, y) // Horizontal Flip
    } else {
        (x, y)
    };

    // Then apply rotation (0, 90, 180, 270)
    // Rotation is Clockwise
    match mode % 4 {
        0 => (tx, ty),             // 0 deg
        1 => (ty, max - tx),       // 90 deg
        2 => (max - tx, max - ty), // 180 deg
        3 => (max - ty, tx),       // 270 deg
        _ => unreachable!(),
    }
}

// Transforms the label (0=Up, 1=Right, 2=Down, 3=Left)
fn transform_label(label: usize, mode: u8) -> usize {
    let mut new_label = label;

    // 1. Handle Flip
    // Flip Horizontal (axis is vertical line):
    // Up(0)->Up(0), Down(2)->Down(2), Left(3)<->Right(1)
    if mode >= 4 {
        if new_label == 1 {
            new_label = 3;
        } else if new_label == 3 {
            new_label = 1;
        }
    }

    // 2. Handle Rotation (Clockwise)
    // Up(0) -> Right(1) -> Down(2) -> Left(3)
    let rotation = (mode % 4) as usize;
    new_label = (new_label + rotation) % 4;

    new_label
}

// Updated encode_board to accept symmetry_mode
fn encode_board(
    item: &TrainingExample,
    my_snake: &Snake,
    my_len: usize,
    buffer: &mut [f32],
    mode: u8, // New argument
) {
    let target_id = &my_snake.id;

    for snake in &item.snakes {
        if snake.death.is_some() {
            continue;
        }

        let len = snake.body.len();
        let is_me = snake.id == *target_id;
        let health_norm = snake.health as f32 / 100.0;

        // HEAD
        let head = &snake.body[0];
        // Note: we check bounds on original, then transform, then check bounds again (implicitly via set_channel)
        if in_bounds(head.x, head.y) {
            let (tx, ty) = transform_pos(head.x, head.y, mode);

            if is_me {
                set_channel(buffer, CH_MY_HEAD, tx, ty, 1.0);
            } else {
                let relative_size = len as f32 / (len + my_len) as f32;
                set_channel(buffer, CH_ENEMY_HEADS, tx, ty, relative_size);
                set_channel(buffer, CH_ENEMY_HEALTH, tx, ty, health_norm);
            }
        }

        // BODY
        if len > 2 {
            for (i, part) in snake.body.iter().enumerate().skip(1).take(len - 2) {
                if !in_bounds(part.x, part.y) {
                    continue;
                }

                let (tx, ty) = transform_pos(part.x, part.y, mode);
                let gradient = (len - i) as f32 / len as f32;

                if is_me {
                    set_channel(buffer, CH_MY_BODY, tx, ty, gradient);
                } else {
                    set_channel(buffer, CH_ENEMY_BODIES, tx, ty, gradient);
                    set_channel(buffer, CH_ENEMY_HEALTH, tx, ty, health_norm);
                }
            }
        }

        // TAIL
        if len > 1 {
            if let Some(tail) = snake.body.last() {
                if in_bounds(tail.x, tail.y) {
                    let (tx, ty) = transform_pos(tail.x, tail.y, mode);

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

    // FOOD
    for food in &item.food {
        if in_bounds(food.x, food.y) {
            let (tx, ty) = transform_pos(food.x, food.y, mode);
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
    // Safety check for transformed coordinates
    if !in_bounds(x, y) {
        return;
    }
    let idx = (channel * GRID_SIZE * GRID_SIZE) + (y as usize * GRID_SIZE) + x as usize;
    buffer[idx] = value;
}
