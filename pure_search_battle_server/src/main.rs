use std::time::Duration;

use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
mod engine;
use engine::{search, GameMoveRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(handle_info))
        .route("/move", post(handle_move));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Pure search battle server listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_move(Json(req): Json<GameMoveRequest>) -> Json<MoveResponse> {
    let result = search(&req, Duration::from_millis(470));
    println!("TURN {} -------", req.turn);
    println!("Best move: {}\n", result.best_move);

    println!("Move scores:");
    for score in &result.move_scores {
        println!(
            "  {:>5}: {:>+8.3} avg score ({} leaves)",
            score.direction.as_str(),
            score.score,
            score.leaf_count
        );
    }

    println!("\nSearch statistics:");
    println!("  Total nodes explored: {}", result.total_nodes);
    println!("  Max depth reached: {}", result.max_depth);
    println!("  Time elapsed: {}ms", result.elapsed_ms);
    println!("  Throughput: {:.2}M nodes/sec", result.throughput_mnps);
    println!("------- END {}\n\n", req.turn);

    Json(MoveResponse {
        r#move: result.best_move.as_str().to_string(),
        shout: "⛓️🌹🌹⛓️".to_string(),
    })
}

async fn handle_info() -> Json<InfoResponse> {
    Json(InfoResponse {
        apiversion: "1".into(),
        color: "#6B3A3A".to_string(),
        head: "rose".to_string(),
        tail: "flytrap".to_string(),
    })
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
