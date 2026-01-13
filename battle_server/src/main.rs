use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array2, Array3, Array4};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use burn::{module::Module, record::{FullPrecisionSettings, NamedMpkFileRecorder}};
use burn::{record::CompactRecorder, record::Recorder, tensor::{Tensor, TensorData}};
use burn_ndarray::NdArray;

use burn_ai_model::{new_cnn::{BattleCnn, BattleCnnConfig}, simple_cnn_opset16::Model as ModelOriginal};
use burn_ai_model::transformer::{BattleModel, BattleModelConfig}; 
use burn_ai_model::transformer_winprob::{BattleModel as WinProbBattleModel, BattleModelConfig as WinProbBattleModelConfig}; 

// DEFINE THE BACKEND
// We use NdArray for pure CPU execution.
// <f32> indicates the float precision.
type B = NdArray<f32>;

// 1. Define a helper enum so we know what to preprocess 
// WITHOUT locking the heavy Mutex first.
#[derive(Clone, Copy, Debug)]
enum ModelKind {
    OriginalCnn,
    HydraTransformer,
    OxTransformer,
    NewHydraCnn,
}
impl ModelKind {
    pub fn color(&self) -> &'static str {
        match self {
            ModelKind::OriginalCnn => "#D34516",
            ModelKind::HydraTransformer => "#D34516",
            ModelKind::OxTransformer => "#1E2650",
            ModelKind::NewHydraCnn => "#6B3A3A"
        }
    }
    pub fn head(&self) -> &'static str { 
        match self { 
            ModelKind::OriginalCnn => "egg",
            ModelKind::HydraTransformer => "cute-dragon",
            ModelKind::OxTransformer => "bull",
            ModelKind::NewHydraCnn => "cute-dragon"
        }
    }
    pub fn tail(&self) -> &'static str { 
        match self { 
            ModelKind::OriginalCnn => "egg",
            ModelKind::HydraTransformer => "duck",
            ModelKind::OxTransformer => "rocket",
            ModelKind::NewHydraCnn => "flytrap"
        }
    }
}

// 2. Wrap State so we can access the Kind cheaply
#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<Model>>,
    kind: ModelKind,
}


enum Model { 
    Original(ModelOriginal<B>),
    Transformer(BattleModel<B>),
    TransformerWinProb(WinProbBattleModel<B>),
    NewHydraCnn(BattleCnn<B>)
}

#[derive(Deserialize)]
struct Position {
    x: usize,
    y: usize,
}

#[derive(Deserialize)]
struct Snake {
    id: String,
    health: u32,
    body: Vec<Position>,
    #[serde(default)]
    length: usize,
}

#[derive(Deserialize)]
struct RulesetSettings {
    #[serde(rename = "foodSpawnChance")]
    food_spawn_chance: u32,
    #[serde(rename = "minimumFood")]
    minimum_food: u32,
}

#[derive(Deserialize)]
struct Ruleset {
    settings: RulesetSettings,
}

#[derive(Deserialize)]
struct Game {
    ruleset: Ruleset,
}

#[derive(Deserialize)]
struct Board {
    height: usize,
    width: usize,
    food: Vec<Position>,
    hazards: Vec<Position>,
    snakes: Vec<Snake>,
}

#[derive(Deserialize)]
struct GameMoveRequest {
    game: Game,
    turn: u32,
    board: Board,
    you: Snake,
}

#[derive(Serialize)]
struct MoveResponse {
    r#move: String,
    shout: String,
}

#[derive(Serialize)]
struct InfoResponse {
    apiversion: String,
    color: String,
    head: String,
    tail: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

const CHANNELS: usize = 8;
const HEIGHT: usize = 11;
const WIDTH: usize = 11;
const METADATA_COUNT: usize = 8;

fn preprocess(req: &GameMoveRequest) -> (Array4<f32>, Array2<f32>) {
    let (w, h) = (req.board.width, req.board.height);
    let area = (w * h) as f32;
    let target_id = &req.you.id;

    let mut board = Array4::<f32>::zeros((1, CHANNELS, HEIGHT, WIDTH));

    let mut other_len_sum = 0usize;
    let mut other_count = 0usize;
    let mut longest_other = 0usize;

    for snake in &req.board.snakes {
        let len = snake.body.len();
        let norm_len = len as f32 / area;
        let norm_health = snake.health as f32 / 100.0;
        let is_target = snake.id == *target_id;

        if !is_target {
            other_len_sum += len;
            other_count += 1;
            longest_other = longest_other.max(len);
        }

        // Head
        let head = &snake.body[0];
        let head_ch = if is_target { 0 } else { 1 };
        board[[0, head_ch, head.y, head.x]] = norm_len;

        // Body (excluding head and tail)
        let body_ch = if is_target { 2 } else { 3 };
        for part in snake
            .body
            .iter()
            .skip(1)
            .take(snake.body.len().saturating_sub(2))
        {
            board[[0, body_ch, part.y, part.x]] = norm_health;
        }

        // Tail
        if let Some(tail) = snake.body.last() {
            let tail_ch = if is_target { 4 } else { 5 };
            board[[0, tail_ch, tail.y, tail.x]] = norm_health;
        }
    }

    // Food
    for f in &req.board.food {
        if f.x < w && f.y < h {
            board[[0, 6, f.y, f.x]] += 1.0;
        }
    }

    // Hazards
    for hz in &req.board.hazards {
        if hz.x < w && hz.y < h {
            board[[0, 7, hz.y, hz.x]] += 1.0;
        }
    }

    // Metadata
    let food_spawn = req.game.ruleset.settings.food_spawn_chance as f32 / 100.0;
    let min_food = req.game.ruleset.settings.minimum_food as f32 / area;
    let turn = req.turn as f32 / 7200.0;
    let head_x = req.you.body[0].x as f32 / w as f32;
    let head_y = req.you.body[0].y as f32 / h as f32;
    let health = req.you.health as f32 / 100.0;
    let target_len = req.you.body.len();
    let is_longest = if target_len > longest_other { 1.0 } else { 0.0 };
    let avg_frac = if is_longest > 0.5 && other_count > 0 {
        (other_len_sum as f32 / other_count as f32) / target_len as f32
    } else {
        0.0
    };

    let metadata = Array2::from_shape_vec(
        (1, METADATA_COUNT),
        vec![
            food_spawn, min_food, turn, head_x, head_y, health, is_longest, avg_frac,
        ],
    )
    .unwrap();

    (board, metadata)
}

// Constants for Transformer
const GRID_SIZE: usize = 11;
const SEQ_LEN: usize = 121;
const TILE_FEATS: usize = 27;
const META_FEATS: usize = 2;

// Helper to set features safely
fn set_feat(grid: &mut [f32], occupied: &mut [bool], x: i32, y: i32, feat: usize, val: f32) {
    if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
        let tile_idx = (y as usize * GRID_SIZE) + x as usize;
        
        // Mark as occupied
        occupied[tile_idx] = true;

        let vec_idx = (tile_idx * TILE_FEATS) + feat;
        grid[vec_idx] = val;
    }
}

fn preprocess_transformer(req: &GameMoveRequest) -> (Array3<f32>, Array2<f32>) {
    let (w, h) = (req.board.width, req.board.height);
    let area = (w * h) as f32;
    
    // 1. Tiles [1, 121, 27]
    let mut grid = vec![0.0f32; SEQ_LEN * TILE_FEATS];
    
    // CHANGED: Initialize occupancy mask
    let mut occupied = vec![false; SEQ_LEN];
    
    let my_id = &req.you.id;
    let mut enemies: Vec<&Snake> = req.board.snakes.iter()
        .filter(|s| s.id != *my_id)
        .collect();
    enemies.sort_by_key(|s| &s.id);

    // Helper to process any snake into the grid
    let mut process_snake = |snake: &Snake, offset: usize| {
        let len = snake.body.len();
        let norm_health = snake.health as f32 / 100.0;
        let norm_len = len as f32 / area;

        for (i, part) in snake.body.iter().enumerate() {
            let x = part.x as i32;
            let y = part.y as i32;

            // Pass &mut occupied to set_feat
            if i == 0 {
                set_feat(&mut grid, &mut occupied, x, y, offset + 0, 1.0); // Head
            } else if i == 1 {
                set_feat(&mut grid, &mut occupied, x, y, offset + 1, 1.0); // Neck
            } else if i == len - 1 {
                set_feat(&mut grid, &mut occupied, x, y, offset + 2, 1.0); // Tail
            } else {
                let turns_remaining = (len - i) as f32;
                let gradient_val = turns_remaining / len as f32;
                set_feat(&mut grid, &mut occupied, x, y, offset + 3, gradient_val);
            }

            set_feat(&mut grid, &mut occupied, x, y, offset + 4, norm_health);
            set_feat(&mut grid, &mut occupied, x, y, offset + 5, norm_len);
        }
    };

    // Fill Me
    process_snake(&req.you, 0);

    // Fill Enemies
    for (i, enemy) in enemies.iter().take(3).enumerate() {
        process_snake(enemy, 6 + (i * 6));
    }

    for f in &req.board.food {
        set_feat(&mut grid, &mut occupied, f.x as i32, f.y as i32, 24, 1.0);
    }
    for hz in &req.board.hazards {
        set_feat(&mut grid, &mut occupied, hz.x as i32, hz.y as i32, 25, 1.0);
    }

    // CHANGED: Final pass for Empty Tiles (Feature Index 26)
    for tile_idx in 0..SEQ_LEN {
        if !occupied[tile_idx] {
            // We calculate index manually since we aren't using x/y here
            let vec_idx = (tile_idx * TILE_FEATS) + 26;
            grid[vec_idx] = 1.0;
        }
    }

    // 2. Metadata [1, 2]
    let fs_chance = req.game.ruleset.settings.food_spawn_chance as f32;
    let min_food = req.game.ruleset.settings.minimum_food as f32;
    
    let meta_vec = vec![
        fs_chance / 100.0,
        min_food / area,
    ];

    let tiles_array = Array3::from_shape_vec((1, SEQ_LEN, TILE_FEATS), grid).unwrap();
    let meta_array = Array2::from_shape_vec((1, META_FEATS), meta_vec).unwrap();

    (tiles_array, meta_array)
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing for BattleCnn (New Model)
// ─────────────────────────────────────────────────────────────────────────────
//
// Channel encoding (8 channels):
// 0: My head        - value: 1.0
// 1: Enemy heads    - value: their_len / (their_len + my_len) [relative size]
// 2: My body        - value: gradient from head (1.0) toward tail (→0)
// 3: Enemy bodies   - value: gradient per-snake
// 4: My tail        - value: 1.0
// 5: Enemy tails    - value: 1.0
// 6: Food           - value: 1.0
// 7: Enemy health   - value: health/100 on ALL enemy tiles (head, body, tail)
//
// Metadata (3 values):
// 0: food_spawn_chance / 100.0
// 1: min_food / board_area
// 2: my_health / 100.0

const CNN_CHANNELS: usize = 8;
const CNN_HEIGHT: usize = 11;
const CNN_WIDTH: usize = 11;
const CNN_META_COUNT: usize = 3;

// Channel indices
const CH_MY_HEAD: usize = 0;
const CH_ENEMY_HEADS: usize = 1;
const CH_MY_BODY: usize = 2;
const CH_ENEMY_BODIES: usize = 3;
const CH_MY_TAIL: usize = 4;
const CH_ENEMY_TAILS: usize = 5;
const CH_FOOD: usize = 6;
const CH_ENEMY_HEALTH: usize = 7;

fn preprocess_cnn(req: &GameMoveRequest) -> (Array4<f32>, Array2<f32>) {
    let (w, h) = (req.board.width, req.board.height);
    let area = (w * h) as f32;
    let my_id = &req.you.id;
    let my_len = req.you.body.len();

    let mut board = Array4::<f32>::zeros((1, CNN_CHANNELS, CNN_HEIGHT, CNN_WIDTH));

    for snake in &req.board.snakes {
        let len = snake.body.len();
        let is_me = snake.id == *my_id;
        let health_norm = snake.health as f32 / 100.0;

        // === HEAD ===
        let head = &snake.body[0];
        if head.y < CNN_HEIGHT && head.x < CNN_WIDTH {
            if is_me {
                // My head: simple marker
                board[[0, CH_MY_HEAD, head.y, head.x]] = 1.0;
            } else {
                // Enemy head: relative size encoding
                // > 0.5 means they're bigger (dangerous for collisions)
                // < 0.5 means I'm bigger (safe to challenge)
                // = 0.5 means equal size
                let relative_size = len as f32 / (len + my_len) as f32;
                board[[0, CH_ENEMY_HEADS, head.y, head.x]] = relative_size;

                // Also mark enemy health on head tile
                board[[0, CH_ENEMY_HEALTH, head.y, head.x]] = health_norm;
            }
        }

        // === BODY (excluding head and tail) ===
        // Gradient encoding: how soon will this tile be empty?
        // Value near 1.0 = just behind head, won't clear for a while
        // Value near 0.0 = close to tail, will clear soon
        if len > 2 {
            for (i, part) in snake.body.iter().enumerate().skip(1).take(len - 2) {
                if part.y >= CNN_HEIGHT || part.x >= CNN_WIDTH {
                    continue;
                }

                // i=1 is first body segment after head
                // gradient = (len - i) / len
                //   i=1: (len-1)/len ≈ 1.0
                //   i=len-2: 2/len (small)
                let gradient = (len - i) as f32 / len as f32;

                if is_me {
                    board[[0, CH_MY_BODY, part.y, part.x]] = gradient;
                } else {
                    board[[0, CH_ENEMY_BODIES, part.y, part.x]] = gradient;
                    board[[0, CH_ENEMY_HEALTH, part.y, part.x]] = health_norm;
                }
            }
        }

        // === TAIL ===
        // Only mark if snake length > 1 (otherwise head == tail)
        if len > 1 {
            if let Some(tail) = snake.body.last() {
                if tail.y < CNN_HEIGHT && tail.x < CNN_WIDTH {
                    if is_me {
                        board[[0, CH_MY_TAIL, tail.y, tail.x]] = 1.0;
                    } else {
                        board[[0, CH_ENEMY_TAILS, tail.y, tail.x]] = 1.0;
                        board[[0, CH_ENEMY_HEALTH, tail.y, tail.x]] = health_norm;
                    }
                }
            }
        }
    }

    // === FOOD ===
    for f in &req.board.food {
        if f.x < w && f.y < h && f.y < CNN_HEIGHT && f.x < CNN_WIDTH {
            board[[0, CH_FOOD, f.y, f.x]] = 1.0;
        }
    }

    // === METADATA ===
    let food_spawn = req.game.ruleset.settings.food_spawn_chance as f32 / 100.0;
    let min_food = req.game.ruleset.settings.minimum_food as f32 / area;
    let my_health = req.you.health as f32 / 100.0;

    let metadata = Array2::from_shape_vec(
        (1, CNN_META_COUNT),
        vec![food_spawn, min_food, my_health],
    )
    .unwrap();

    (board, metadata)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn handle_info(State(state): State<AppState>) -> Json<InfoResponse> {
    let (color, head, tail) = { 
        let model_kind = state.kind;        
        (model_kind.color(), model_kind.head(), model_kind.tail())
    };

    Json(InfoResponse {
        apiversion: "1".into(),
        color: color.into(),
        head: head.into(),
        tail: tail.into(),
    })
}

enum PreprocessedData {
    Cnn { board: Array4<f32>, meta: Array2<f32> },
    Transformer { tiles: Array3<f32>, meta: Array2<f32> },
}

fn to_tensors(b: Array4<f32>, m: Array2<f32>, device: &burn_ndarray::NdArrayDevice) -> (Tensor<B, 4>, Tensor<B, 2>) {
    // Fix: Get shape BEFORE consuming the array into a vector
    let b_shape = b.shape().to_vec();
    let b_vec = b.into_raw_vec();
    let b_tensor = Tensor::from_data(TensorData::new(b_vec, b_shape), device);

    let m_shape = m.shape().to_vec();
    let m_vec = m.into_raw_vec();
    let m_tensor = Tensor::from_data(TensorData::new(m_vec, m_shape), device);

    (b_tensor, m_tensor)
}


async fn handle_move(
    State(state): State<AppState>,
    Json(req): Json<GameMoveRequest>,
) -> Json<MoveResponse> {

    let model_kind = state.kind; 

    let (best_move, shout) = tokio::task::spawn_blocking(move || {
        // 1. CPU WORK (No Lock)
        let input_data = match model_kind {
            ModelKind::OriginalCnn => {
                let (b, m) = preprocess(&req); 
                PreprocessedData::Cnn { board: b, meta: m }
            },
            ModelKind::NewHydraCnn => {
                let (b, m) = preprocess_cnn(&req);  // <-- Our new function
                PreprocessedData::Cnn { board: b, meta: m }
            },
            ModelKind::HydraTransformer => {
                let (t, m) = preprocess_transformer(&req); 
                PreprocessedData::Transformer { tiles: t, meta: m }
            },
            ModelKind::OxTransformer => { 
                let (t, m) = preprocess_transformer(&req); 
                PreprocessedData::Transformer { tiles: t, meta: m }
            }
        };

        // 2. INFERENCE (Lock)
        let model = state.model.lock().expect("mutex poisoned");
        let device = Default::default();

        let output = match (&*model, input_data) {
            // TRANSFORMER CASE
            (Model::Transformer(m), PreprocessedData::Transformer { tiles, meta }) => {
                // Fix: Get shapes BEFORE into_raw_vec
                let t_shape = tiles.shape().to_vec();
                let t_vec = tiles.into_raw_vec();
                
                let m_shape = meta.shape().to_vec();
                let m_vec = meta.into_raw_vec();

                let t_tensor = Tensor::<B, 3>::from_data(
                    TensorData::new(t_vec, t_shape), 
                    &device
                );
                let m_tensor = Tensor::<B, 2>::from_data(
                    TensorData::new(m_vec, m_shape), 
                    &device
                );
                m.forward(t_tensor, m_tensor)
            },
            (Model::TransformerWinProb(m), PreprocessedData::Transformer { tiles, meta }) => { 
                                let t_shape = tiles.shape().to_vec();
                let t_vec = tiles.into_raw_vec();
                
                let m_shape = meta.shape().to_vec();
                let m_vec = meta.into_raw_vec();

                let t_tensor = Tensor::<B, 3>::from_data(
                    TensorData::new(t_vec, t_shape), 
                    &device
                );
                let m_tensor = Tensor::<B, 2>::from_data(
                    TensorData::new(m_vec, m_shape), 
                    &device
                );
                m.forward(t_tensor, m_tensor)
            },
            // CNN CASES
            (Model::Original(m), PreprocessedData::Cnn { board, meta }) => {
                let (b, m_tens) = to_tensors(board, meta, &device);
                m.forward(b, m_tens)
            },
            (Model::NewHydraCnn(m), PreprocessedData::Cnn { board, meta }) => {
                let (b, m_tens) = to_tensors(board, meta, &device);
                m.forward(b, m_tens)
            }
            _ => panic!("Model / Data Mismatch!"),
        };

        println!("Turn: {:?} | Probs: {:?}", req.turn , output);
        let best_idx = output.argmax(1).into_scalar() as usize;
        let moves = match model_kind { 
            ModelKind::OriginalCnn =>  ["up", "right", "down", "left"],
            ModelKind::HydraTransformer =>  ["up", "right", "down", "left"],
            ModelKind::NewHydraCnn => ["up", "right", "down", "left"],
            ModelKind::OxTransformer => ["up","down","right","left"],
        };
        (moves[best_idx].to_string(), "🔥".to_string())
    })
    .await
    .expect("Blocking task failed");

    Json(MoveResponse { r#move: best_move, shout })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let device = burn_ndarray::NdArrayDevice::Cpu;

    let model_choice = std::env::var("WHICH_MODEL")
        .inspect_err(|e| { 
            println!("Error finding the value via 'WHICH_MODEL' env var; falling back to simple_cnn_opset16 | {}", e)
        })
        .inspect(|which_model| {
            println!("Loading model via 'WHICH_MODEL' env var choice: {}", which_model)
        })
        .unwrap_or_else(|_| "simple_cnn_opset16".to_string());

    let (model_enum, kind) = match model_choice.as_str() {
        "simple_cnn_opset16" => { 
            let m = ModelOriginal::from_file("simple_cnn_opset16", &device);
            (Model::Original(m), ModelKind::OriginalCnn)
        },
        "transformer_scout" => {
             // MUST MATCH TRAINING CONFIG!
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

             
             // Load the record explicitly
             let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                .load("model-70".into(), &device)
                .expect("Failed to load transformer weights");
             
             // Init and load
             let model = BattleModel::new(&config, &device).load_record(record);
             
             (Model::Transformer(model), ModelKind::HydraTransformer)
        },
        "value_model" => { 
            let config = WinProbBattleModelConfig {
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

             
             // Load the record explicitly
             let record: burn_ai_model::transformer_winprob::BattleModelRecord<NdArray> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                .load("ox-model-10".into(), &device)
                .expect("Failed to load transformer weights");
            //not sure what epoch that landed on but it was more than the 21 in there. I *think* still in the 20s though, like maybe 26.
            //also no guarantee it was actually improved, though loss seemed to be steadily dropping just fine, so probably ok.
             
             // Init and load
             let model = WinProbBattleModel::new(&config, &device).load_record(record);
             
             (Model::TransformerWinProb(model), ModelKind::OxTransformer)
        }
        "new_cnn" => { 
            let config = BattleCnnConfig::new();
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                .load("model-12".into(), &device)
                .expect("Failed to load new CNN weights");
            let model = BattleCnn::new(&config, &device).load_record(record);
            (Model::NewHydraCnn(model), ModelKind::NewHydraCnn)
        },
        "new_cnn_experimental" => { 
            let config = BattleCnnConfig::new();
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                .load("model-10".into(), &device)
                .expect("Failed to load new CNN weights");
            let model = BattleCnn::new(&config, &device).load_record(record);
            (Model::NewHydraCnn(model), ModelKind::NewHydraCnn)
        },
        _ => {
            println!("Unrecognized model choice, falling back to simple_cnn_opset16");
            let m = ModelOriginal::from_file("simple_cnn_opset16", &device);
            (Model::Original(m), ModelKind::OriginalCnn)
        }
    };


    let state = AppState {
        model: Arc::new(Mutex::new(model_enum)),
        kind,
    };

    let app = Router::new()
        .route("/", get(handle_info))
        .route("/move", post(handle_move))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    Ok(())
}
