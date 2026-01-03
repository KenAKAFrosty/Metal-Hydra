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
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE;
const TILE_FEATS: usize = 26;
const META_FEATS: usize = 2;

// Total floats per example: (121 * 26) + 2 + 1 (label)
const FLOATS_PER_RECORD: usize = (SEQ_LEN * TILE_FEATS) + META_FEATS + 1;

fn main() {
    let db_path = "../battlesnake_data.db"; // Adjust if needed
    let out_path = "train_data.bin";

    println!("Loading data from {}...", db_path);
    let conn = Connection::open(db_path).expect("Could not open DB");
    // We use a fast query
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
    // Re-use this buffer to avoid allocation
    let mut buffer = vec![0.0f32; FLOATS_PER_RECORD];

    println!("Processing and writing records...");
    for (i, item) in rows.enumerate() {
        if i % 10_000 == 0 {
            print!("\rProcessed {}...", i);
            std::io::stdout().flush().unwrap();
        }
        let item = item.unwrap();

        // Zero out the buffer for the new game state
        buffer.fill(0.0);

        let w = item.game.width as usize;
        let h = item.game.height as usize;
        let area = (w * h) as f32;

        // --- CLOSURE: Write a specific feature to the flat buffer ---
        let mut write_feature = |x: i32, y: i32, feat_offset: usize, val: f32, buf: &mut [f32]| {
            if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
                let tile_idx = (y as usize * GRID_SIZE) + x as usize;
                let global_idx = (tile_idx * TILE_FEATS) + feat_offset;
                buf[global_idx] = val;
            }
        };

        // --- LOGIC: Snake Processing (Matches your original Batcher) ---
        let target_id = &item.winning_snake_id;
        let mut my_snake: Option<Snake> = None;
        let mut enemies: Vec<Snake> = Vec::new();

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

        let mut process_snake = |snake: &Snake, offset: usize, buf: &mut [f32]| {
            let len = snake.body.len();
            let norm_health = snake.health as f32 / 100.0;
            let norm_len = len as f32 / area;

            for (i, part) in snake.body.iter().enumerate() {
                // 1. Anchors
                if i == 0 {
                    write_feature(part.x, part.y, offset + 0, 1.0, buf); // Head
                } else if i == 1 {
                    write_feature(part.x, part.y, offset + 1, 1.0, buf); // Neck
                } else if i == len - 1 {
                    write_feature(part.x, part.y, offset + 2, 1.0, buf); // Tail
                } else {
                    // Body Gradient
                    let turns_remaining = (len - i) as f32;
                    let gradient_val = turns_remaining / len as f32;
                    write_feature(part.x, part.y, offset + 3, gradient_val, buf);
                }

                // 2. Holographic Stats
                write_feature(part.x, part.y, offset + 4, norm_health, buf);
                write_feature(part.x, part.y, offset + 5, norm_len, buf);
            }
        };

        // 1. My Snake (Offset 0)
        if let Some(me) = my_snake {
            process_snake(&me, 0, &mut buffer);
        }

        // 2. Enemies (Offset 6, 12, 18)
        enemies.sort_by(|a, b| a.id.cmp(&b.id));
        for (i, enemy) in enemies.iter().take(3).enumerate() {
            let offset = 6 + (i * 6);
            process_snake(enemy, offset, &mut buffer);
        }

        // 3. Global Features
        for food in item.food {
            write_feature(food.x, food.y, 24, 1.0, &mut buffer);
        }
        for hazard in item.hazards {
            write_feature(hazard.x, hazard.y, 25, 1.0, &mut buffer);
        }

        // --- METADATA & LABEL ---
        let meta_start = SEQ_LEN * TILE_FEATS; // Index where grid ends

        let food_chance = item
            .game
            .ruleset
            .food_spawn_chance
            .parse::<f32>()
            .unwrap_or(15.0);
        let min_food = item.game.ruleset.minimum_food.parse::<f32>().unwrap_or(1.0);

        buffer[meta_start] = food_chance / 100.0;
        buffer[meta_start + 1] = min_food / area;

        // Label (Stored as f32, cast back to int during training)
        buffer[meta_start + 2] = item.label as f32;

        // Write raw bytes
        let bytes = bytemuck::cast_slice(&buffer);
        writer.write_all(bytes).unwrap();

        count += 1;
    }

    // Update the header count
    writer.flush().unwrap();
    let mut f = writer.into_inner().unwrap();
    f.seek(std::io::SeekFrom::Start(0)).unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();

    println!(
        "\nDone! Wrote {} records to {}. Size: {:.2} GB",
        count,
        out_path,
        (count as usize * FLOATS_PER_RECORD * 4) as f32 / 1e9
    );
}
