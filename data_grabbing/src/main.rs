use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio_tungstenite::connect_async;
use url::Url;
use futures::stream::{self, StreamExt};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Arc;

// --- Type Definitions ---

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "PascalCase")]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
struct DeathInfo {
    turn: u32,
    cause: String,
    eliminated_by: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
struct Snake {
    #[serde(rename = "ID")]
    id: String,
    body: Vec<Position>,
    health: i32,
    is_bot: bool,
    is_environment: bool,
    death: Option<DeathInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
struct BoardState {
    turn: u32,
    snakes: Vec<Snake>,
    food: Vec<Position>,
    hazards: Vec<Position>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")] // Note: Ruleset usually uses camelCase in API
struct Ruleset {
    food_spawn_chance: String,
    minimum_food: String,
    name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
struct Game {
    #[serde(rename = "ID")]
    id: String,
    status: String,
    width: u32,
    height: u32,
    ruleset: Ruleset,
    snake_timeout: u32,
    max_turns: u32,
}

// Initial HTTP Response Wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GameInfoResponse {
    game: Game,
    last_frame: BoardState,
}

// WebSocket Message Wrapper
#[derive(Debug, Deserialize)]
struct WsEvent {
    #[serde(rename = "Type")]
    event_type: String, // "frame" or "game_end"
    #[serde(rename = "Data")]
    data: BoardState,
}

// Final Training Data Structure
#[derive(Debug, Serialize)]
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
    label: u8,
    winning_snake_id: String,
}

// --- Logic Helper ---

fn find_position_moved(before: &Position, after: &Position) -> u8 {
    // 0: Up, 1: Right, 2: Down, 3: Left
    if after.x == before.x {
        if after.y > before.y {
            0 // Up
        } else {
            2 // Down
        }
    } else {
        if after.x > before.x {
            1 // Right
        } else {
            3 // Left
        }
    }
}

// --- Main Function ---

pub async fn get_game_training_data(
    game_id: &str,
) -> Result<Option<Vec<TrainingExample>>, Box<dyn Error + Send + Sync>> {
    let http_client = Client::new();

    // 1. Fetch Game Info via HTTP
    let game_info_url = format!("https://engine.battlesnake.com/games/{}", game_id);
    let resp: GameInfoResponse = http_client.get(&game_info_url).send().await?.json().await?;

    // 2. Determine Winner from LastFrame
    let survivors: Vec<&Snake> = resp
        .last_frame
        .snakes
        .iter()
        .filter(|s| s.death.is_none())
        .collect();

    if survivors.len() != 1 {
        // Draw or everyone died
        return Ok(None);
    }
    let winning_snake_id = survivors[0].id.clone();

    // 3. Connect to WebSocket
    let ws_url = format!("wss://engine.battlesnake.com/games/{}/events", game_id);
    let (ws_stream, _) = connect_async(Url::parse(&ws_url)?).await?;

    // We only care about the read part of the stream
    let (_, mut read) = ws_stream.split();

    let mut frames: Vec<BoardState> = Vec::new();

    println!("WebSocket opened for {}", game_id);

    // 4. Stream Processing
    // In Rust, we iterate the stream until it ends (None) which signifies "close"
    while let Some(message) = read.next().await {
        match message {
            Ok(msg) => {
                if let Ok(text) = msg.to_text() {
                    // Ignore parsing errors for keepalives or non-JSON messages
                    if let Ok(event) = serde_json::from_str::<WsEvent>(text) {
                        if event.event_type == "frame" {
                            frames.push(event.data);
                        }
                    }
                }
            }
            Err(e) => eprintln!("Error reading message: {}", e),
        }
    }

    println!("WebSocket closed. Frames collected: {}", frames.len());

    // 5. Sort Frames
    frames.sort_by_key(|f| f.turn);

    // 6. Generate Training Examples
    let mut training_examples = Vec::new();

    // Use .windows(2) to iterate over [frame[i-1], frame[i]]
    for window in frames.windows(2) {
        let last_frame = &window[0];
        let this_frame = &window[1];

        // Find the winning snake in both frames
        let snake_curr = this_frame.snakes.iter().find(|s| s.id == winning_snake_id);
        let snake_prev = last_frame.snakes.iter().find(|s| s.id == winning_snake_id);

        if let (Some(curr), Some(prev)) = (snake_curr, snake_prev) {
            // Guard against empty bodies (shouldn't happen in valid games)
            if curr.body.is_empty() || prev.body.is_empty() {
                continue;
            }

            let head_curr = &curr.body[0];
            let head_prev = &prev.body[0];

            let label = find_position_moved(head_prev, head_curr);

            training_examples.push(TrainingExample {
                game: resp.game.clone(),
                snakes: last_frame.snakes.clone(),
                food: last_frame.food.clone(),
                hazards: last_frame.hazards.clone(),
                turn: last_frame.turn,
                label,
                winning_snake_id: winning_snake_id.clone(),
            });
        }
    }

    println!("Training examples generated: {}", training_examples.len());
    Ok(Some(training_examples))
}

const DB_PATH: &str = "../battlesnake_data.db";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // 1. Initialize Database
    {
        // Run blocking DB init in a separate task
        let _ = tokio::task::spawn_blocking(|| init_db(DB_PATH)).await??;
    }

    // 2. Step A: Scrape Leaderboard for New IDs
    println!("--- Step 1a: Scraping Standard Leaderboard ---");
    scrape_leaderboard_ids(GameMode::Standard).await?;

    println!("--- Step 1b: Scraping Duel Leaderboard ---");
    scrape_leaderboard_ids(GameMode::Duel).await?;

    // 3. Step B: Process Pending Games
    println!("--- Step 2: Processing Pending Games ---");
    process_pending_games().await?;

    Ok(())
}

fn init_db(path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = Connection::open(path)?;
    
    // Enable WAL mode for better concurrency
    conn.pragma_update(None,"journal_mode", "WAL")?;

    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            player_found_via TEXT,
            status TEXT NOT NULL DEFAULT 'pending', 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        "#,
        [],
    )?;

    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS training_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            turn INTEGER NOT NULL,
            label INTEGER NOT NULL,
            data_json TEXT NOT NULL,
            FOREIGN KEY(game_id) REFERENCES games(id)
        );
        "#,
        [],
    )?;

    Ok(())
}

// --- Step 1: Scraping Logic ---

async fn scrape_leaderboard_ids(game_mode: GameMode) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();

    let players = get_top_player_names(&client, 2, game_mode.clone()).await?;
    println!("Found {} players on leaderboard.", players.len());

    let db_path = DB_PATH.to_string();

    // Limit concurrency to avoid file lock contention
    let mut stream = stream::iter(players)
        .map(|player| {
            let c = client.clone();
            let path = db_path.clone();
            let game_mode_copy = game_mode.clone();
            async move {
                match get_games_from_player(&c, &player, game_mode_copy).await {
                    Ok(ids) => {
                        // Blocking DB Insert
                        let player_copy = player.clone();
                        let count = tokio::task::spawn_blocking(move || {
                            insert_game_ids(&path, &player_copy, ids)
                        }).await.unwrap_or(0);
                        
                        println!("Player {}: stored {} new games.", &player, count);
                    }
                    Err(e) => eprintln!("Failed to scrape {}: {}", player, e),
                }
            }
        })
        .buffer_unordered(5); 

    while stream.next().await.is_some() {}
    Ok(())
}

fn insert_game_ids(path: &str, player: &str, ids: Vec<String>) -> usize {
    let conn = match Connection::open(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to open DB for {}: {}", player, e);
            return 0;
        }
    };

    let mut new_count = 0;
    for id in ids {
        let res = conn.execute(
            "INSERT OR IGNORE INTO games (id, player_found_via) VALUES (?1, ?2)",
            params![id, player],
        );
        match res {
            Ok(affected) => {
                if affected > 0 {
                    new_count += 1;
                }
            }
            Err(e) => eprintln!("Insert error for {}: {}", id, e), // <--- SEE THE ERROR
        }
    }
    new_count
}

#[derive(Clone)]
enum GameMode { 
    Standard,
    Duel
}

async fn get_top_player_names(client: &Client, count: usize, game_mode: GameMode) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let suffix = match game_mode { 
        GameMode::Standard => "standard",
        GameMode::Duel => "standard-duels"
    };
    let url = format!("https://play.battlesnake.com/leaderboard/{}", suffix);
    let html = client.get(url).send().await?.text().await?;
    let re = Regex::new(&format!(r"/leaderboard/{}/([^/]+)/stats", suffix))?;
    Ok(re.captures_iter(&html).map(|c| c[1].to_string()).take(count).collect())
}

async fn get_games_from_player(client: &Client, name: &str, game_mode: GameMode) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let suffix = match game_mode { 
        GameMode::Standard => "standard",
        GameMode::Duel => "standard-duels"
    };
    let url = format!("https://play.battlesnake.com/leaderboard/{}/{}/stats",suffix, name);
    let html = client.get(&url).send().await?.text().await?;
    let re = Regex::new(r"/game/([a-z0-9-]+)")?;
    Ok(re.captures_iter(&html).map(|c| c[1].to_string()).collect())
}

// --- Step 2: Processing Logic ---

async fn process_pending_games() -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        let path = DB_PATH.to_string();
        
        // 1. Fetch pending IDs (Blocking)
        let pending_ids = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(path)?;
            let mut stmt = conn.prepare("SELECT id FROM games WHERE status = 'pending' LIMIT 50")?;
            let ids: Result<Vec<String>, _> = stmt.query_map([], |row| row.get(0))?.collect();
            ids
        }).await??;

        if pending_ids.is_empty() {
            println!("No more pending games.");
            break;
        }

        println!("Processing batch of {} games...", pending_ids.len());

        // 2. Process in Parallel
        let mut stream = stream::iter(pending_ids)
            .map(|id| {
                let path = DB_PATH.to_string();
                async move {
                    process_single_game(&path, &id).await
                }
            })
            .buffer_unordered(10);

        while stream.next().await.is_some() {}
    }
    Ok(())
}



async fn process_single_game(db_path: &str, game_id: &str) {
    // A. Network Heavy Lifting (Async)
    let result = get_game_training_data(game_id).await;

    // B. Database Writing (Blocking)
    let db_path = db_path.to_string();
    let game_id = game_id.to_string();

    let _ = tokio::task::spawn_blocking(move || {
        let mut conn = match Connection::open(db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DB Error: {}", e);
                return;
            }
        };

        let tx = match conn.transaction() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Tx Error: {}", e);
                return;
            }
        };

        match result {
            Ok(Some(examples)) => {
                // --- FIX START ---
                // We create a scope { } here. 
                // `stmt` lives inside this scope and is automatically Dropped 
                // at the closing brace `}`.
                {
                    let mut stmt = tx.prepare(
                        "INSERT INTO training_examples (game_id, turn, label, data_json) VALUES (?1, ?2, ?3, ?4)"
                    ).unwrap();
    
                    for ex in examples {
                        let json = serde_json::to_string(&ex).unwrap();
                        let _ = stmt.execute(params![game_id, ex.turn, ex.label, json]);
                    }
                } 
                // --- FIX END ---
                // Now that `stmt` is dropped, `tx` is no longer borrowed, 
                // so we are allowed to consume it with commit().

                let _ = tx.execute("UPDATE games SET status = 'processed' WHERE id = ?1", params![game_id]);
                let _ = tx.commit();
                println!("Processed {}: Saved.", game_id);
            }
            Ok(None) => {
                let _ = tx.execute("UPDATE games SET status = 'invalid' WHERE id = ?1", params![game_id]);
                let _ = tx.commit();
                println!("Processed {}: Invalid/Draw.", game_id);
            }
            Err(e) => {
                eprintln!("Error fetching {}: {}", game_id, e);
                let _ = tx.execute("UPDATE games SET status = 'error' WHERE id = ?1", params![game_id]);
                let _ = tx.commit();
            }
        }
    }).await;
}