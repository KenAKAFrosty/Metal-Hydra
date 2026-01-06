use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

// --- DATA STRUCTURES ---
#[derive(Debug, Clone, Deserialize, Serialize, Copy)]
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
    // Note: The old 'label' field in the JSON is ignored now
    // label: usize,
    winning_snake_id: String,
}

// --- CONSTANTS ---
const GRID_SIZE: usize = 11;
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE;
const TILE_FEATS: usize = 27; // 26 + IsEmpty
const META_FEATS: usize = 2;

// (SEQ_LEN * TILE_FEATS) + META_FEATS + 4 (Values per direction)
const FLOATS_PER_RECORD: usize = (SEQ_LEN * TILE_FEATS) + META_FEATS + 4;

fn main() {
    let db_path = "../battlesnake_data.db";
    let out_path = "train_data_value.bin";

    println!("Loading data from {}...", db_path);
    let conn = Connection::open(db_path).expect("Could not open DB");

    // CRITICAL: We must ORDER BY game_id and turn to group games together
    let mut stmt = conn
        .prepare("SELECT game_id, data_json FROM training_examples ORDER BY game_id ASC, turn ASC")
        .expect("Table not found");

    let rows = stmt
        .query_map([], |row| {
            let game_id: String = row.get(0)?;
            let json: String = row.get(1)?;
            let ex = serde_json::from_str::<TrainingExample>(&json).unwrap();
            Ok((game_id, ex))
        })
        .expect("Query failed");

    let f = File::create(out_path).expect("Unable to create file");
    let mut writer = BufWriter::new(f);
    writer.write_all(&0u64.to_le_bytes()).unwrap(); // Header placeholder

    let mut count: u64 = 0;

    // --- STATE FOR BATCHING ---
    let mut current_game_id = String::new();
    let mut game_buffer: Vec<TrainingExample> = Vec::new();

    // Reusable buffer for writing binary data
    let mut write_buffer = vec![0.0f32; FLOATS_PER_RECORD];

    println!("Processing games...");

    for (i, row) in rows.enumerate() {
        if i % 10_000 == 0 {
            print!("\rProcessed {} DB rows...", i);
            std::io::stdout().flush().unwrap();
        }

        let (row_id, example) = row.unwrap();

        if row_id != current_game_id {
            // New game detected! Process the previous buffer.
            if !game_buffer.is_empty() {
                process_game_buffer(&game_buffer, &mut writer, &mut write_buffer, &mut count);
            }
            // Reset
            current_game_id = row_id;
            game_buffer.clear();
        }

        game_buffer.push(example);
    }

    // Process the final game left in buffer
    if !game_buffer.is_empty() {
        process_game_buffer(&game_buffer, &mut writer, &mut write_buffer, &mut count);
    }

    // Update Header
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

const WINNER_START: f32 = 0.5;
const WINNER_END: f32 = 1.0;
const DEATH_PENALTY: f32 = -1.0;
// The value assigned to actions NOT taken.
// 0.0 is safe. -0.1 would aggressively discourage deviation.
const UNTAKEN_VAL: f32 = 0.0;

fn process_game_buffer(
    turns: &[TrainingExample],
    writer: &mut BufWriter<File>,
    buffer: &mut [f32],
    count: &mut u64,
) {
    if turns.len() < 2 {
        return;
    }

    let winner_id = &turns[0].winning_snake_id;
    let last_turn_index = turns.last().unwrap().turn;

    for window in turns.windows(2) {
        let current_frame = &window[0];
        let next_frame = &window[1];

        for snake in &current_frame.snakes {
            // 1. Skip if dead
            if snake.death.is_some() {
                continue;
            }

            // 2. Find in next frame
            if let Some(next_s) = next_frame.snakes.iter().find(|s| s.id == snake.id) {
                let head_curr = snake.body[0];
                let head_next = next_s.body[0];

                // 3. Static head check (Safety against invalid data)
                if head_curr.x == head_next.x && head_curr.y == head_next.y {
                    continue;
                }

                // 4. Calculate Logic
                let taken_move_idx = get_move_index(head_curr, head_next) as usize;

                // Initialize target vector with the baseline (e.g. 0.0)
                let mut target_vector = [UNTAKEN_VAL; 4];
                let mut should_write = false;

                if snake.id == *winner_id {
                    // --- WINNER ---
                    let current_turn = current_frame.turn as f32;
                    let total_turns = last_turn_index as f32;

                    let ratio = if total_turns > 0.0 {
                        current_turn / total_turns
                    } else {
                        1.0
                    };

                    // Linear ramp: 0.5 -> 1.0
                    let val = WINNER_START + (ratio * (WINNER_END - WINNER_START));

                    // Assign to the specific move taken
                    if taken_move_idx < 4 {
                        target_vector[taken_move_idx] = val;
                        should_write = true;
                    }
                } else {
                    // --- LOSER ---
                    if next_s.death.is_some() {
                        // Immediate Death = -1.0
                        if taken_move_idx < 4 {
                            target_vector[taken_move_idx] = DEATH_PENALTY;
                            should_write = true;
                        }
                    }
                    // Note: We filter out surviving loser moves entirely
                }

                if should_write {
                    // Important: You need to update `write_record_to_buffer` signature
                    // to accept `target_vector: [f32; 4]` instead of `action` and `value`.
                    write_record_to_buffer(current_frame, snake, buffer, target_vector);
                    writer.write_all(bytemuck::cast_slice(buffer)).unwrap();
                    *count += 1;
                }
            }
        }
    }
}

fn write_record_to_buffer(
    item: &TrainingExample,
    focus_snake: &Snake,
    buffer: &mut [f32],
    target_vector: [f32; 4], // <--- CHANGED: Now takes the full vector
) {
    buffer.fill(0.0);
    let mut occupied = [false; SEQ_LEN];
    let area = (item.game.width * item.game.height) as f32;

    // --- CLOSURE: Write features ---
    let mut write_feature =
        |x: i32, y: i32, feat_offset: usize, val: f32, buf: &mut [f32], mask: &mut [bool]| {
            if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
                let tile_idx = (y as usize * GRID_SIZE) + x as usize;
                mask[tile_idx] = true;
                let global_idx = (tile_idx * TILE_FEATS) + feat_offset;
                buf[global_idx] = val;
            }
        };

    // --- Snake Processing ---
    let mut enemies: Vec<&Snake> = item
        .snakes
        .iter()
        .filter(|s| s.id != focus_snake.id && s.death.is_none())
        .collect();

    enemies.sort_by(|a, b| a.id.cmp(&b.id));

    let mut process_snake_body = |s: &Snake, offset: usize, buf: &mut [f32], mask: &mut [bool]| {
        let len = s.body.len();
        let norm_health = s.health as f32 / 100.0;
        let norm_len = len as f32 / area;

        for (i, part) in s.body.iter().enumerate() {
            if i == 0 {
                write_feature(part.x, part.y, offset + 0, 1.0, buf, mask);
            } else if i == 1 {
                write_feature(part.x, part.y, offset + 1, 1.0, buf, mask);
            } else if i == len - 1 {
                write_feature(part.x, part.y, offset + 2, 1.0, buf, mask);
            } else {
                let grad = (len - i) as f32 / len as f32;
                write_feature(part.x, part.y, offset + 3, grad, buf, mask);
            }
            write_feature(part.x, part.y, offset + 4, norm_health, buf, mask);
            write_feature(part.x, part.y, offset + 5, norm_len, buf, mask);
        }
    };

    // 1. Focus Snake
    process_snake_body(focus_snake, 0, buffer, &mut occupied);

    // 2. Enemies
    for (i, enemy) in enemies.iter().take(3).enumerate() {
        let offset = 6 + (i * 6);
        process_snake_body(enemy, offset, buffer, &mut occupied);
    }

    // 3. Global
    for food in &item.food {
        write_feature(food.x, food.y, 24, 1.0, buffer, &mut occupied);
    }
    for hazard in &item.hazards {
        write_feature(hazard.x, hazard.y, 25, 1.0, buffer, &mut occupied);
    }

    // 4. Empty Tiles
    for tile_idx in 0..SEQ_LEN {
        if !occupied[tile_idx] {
            let global_idx = (tile_idx * TILE_FEATS) + 26;
            buffer[global_idx] = 1.0;
        }
    }

    // --- Metadata ---
    let meta_start = SEQ_LEN * TILE_FEATS;
    let food_chance = item
        .game
        .ruleset
        .food_spawn_chance
        .parse::<f32>()
        .unwrap_or(15.0);
    let min_food = item.game.ruleset.minimum_food.parse::<f32>().unwrap_or(1.0);

    buffer[meta_start] = food_chance / 100.0;
    buffer[meta_start + 1] = min_food / area;

    // --- Targets (The 4 Values) ---
    // Make sure these indices match your Batcher logic!
    // 0: Up, 1: Down, 2: Right, 3: Left
    buffer[meta_start + 2] = target_vector[0];
    buffer[meta_start + 3] = target_vector[1];
    buffer[meta_start + 4] = target_vector[2];
    buffer[meta_start + 5] = target_vector[3];
}

fn get_move_index(curr: Position, next: Position) -> f32 {
    let dx = next.x - curr.x;
    let dy = next.y - curr.y;
    match (dx, dy) {
        (0, 1) => 0.0,  // Up
        (0, -1) => 1.0, // Down
        (1, 0) => 2.0,  // Right
        (-1, 0) => 3.0, // Left
        _ => 0.0,
    }
}
