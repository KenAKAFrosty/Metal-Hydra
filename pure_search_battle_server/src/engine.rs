//! High-performance Battlesnake game engine optimized for tree search.
//!
//! This engine is designed for:
//! - Minimal memory footprint (160 bytes per game state)
//! - Fast cloning (all stack-allocated)
//! - O(1) move application
//! - Cache-friendly memory layout
//!
//! # Example
//! ```
//! use battlesnake_engine::{search, GameMoveRequest};
//! use std::time::Duration;
//!
//! let json = r#"{ ... }"#;  // API request JSON
//! let request: GameMoveRequest = serde_json::from_str(json).unwrap();
//! let result = search(&amp;request, Duration::from_millis(450));
//! println!("Best move: {}", result.best_move);
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// =============================================================================
// Constants
// =============================================================================

pub const BOARD_SIZE: usize = 11;
pub const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 121
pub const MAX_SNAKES: usize = 4;
pub const MAX_HEALTH: u8 = 100;
pub const MAX_FOOD: usize = 16;

// Grid cell special values
const EMPTY: u8 = 0;
const FOOD: u8 = 255;
const SENTINEL: u8 = 254;

// =============================================================================
// Direction
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[repr(u8)]
pub enum Direction {
    #[default]
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Direction {
    #[inline(always)]
    pub const fn all() -> [Direction; 4] {
        [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Up => "up",
            Direction::Down => "down",
            Direction::Left => "left",
            Direction::Right => "right",
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Position helpers
// =============================================================================

#[inline(always)]
pub const fn pos_to_idx(x: u8, y: u8) -> u8 {
    y * BOARD_SIZE as u8 + x
}

#[inline(always)]
pub const fn idx_to_pos(idx: u8) -> (u8, u8) {
    (idx % BOARD_SIZE as u8, idx / BOARD_SIZE as u8)
}

#[inline(always)]
pub fn move_pos(idx: u8, dir: Direction) -> Option<u8> {
    let (x, y) = idx_to_pos(idx);
    match dir {
        Direction::Up => {
            if y < (BOARD_SIZE - 1) as u8 {
                Some(pos_to_idx(x, y + 1))
            } else {
                None
            }
        }
        Direction::Down => {
            if y > 0 {
                Some(pos_to_idx(x, y - 1))
            } else {
                None
            }
        }
        Direction::Left => {
            if x > 0 {
                Some(pos_to_idx(x - 1, y))
            } else {
                None
            }
        }
        Direction::Right => {
            if x < (BOARD_SIZE - 1) as u8 {
                Some(pos_to_idx(x + 1, y))
            } else {
                None
            }
        }
    }
}

// =============================================================================
// ArrayVec (stack-allocated dynamic array)
// =============================================================================

#[derive(Clone, Copy)]
pub struct ArrayVec<T: Copy, const N: usize> {
    data: [T; N],
    len: usize,
}

impl<T: Copy + Default, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> ArrayVec<T, N> {
    pub fn new() -> Self {
        ArrayVec {
            data: [T::default(); N],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        if self.len < N {
            self.data[self.len] = value;
            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data[..self.len].iter()
    }
}

impl<T: Copy + Default, const N: usize> std::ops::Index<usize> for ArrayVec<T, N> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

// =============================================================================
// Core Game State (160 bytes, stack-allocated)
// =============================================================================

#[derive(Clone, Copy)]
pub struct GameState {
    /// Grid cells: EMPTY (0), FOOD (255), or pointer toward head
    pub grid: [u8; BOARD_CELLS],
    pub heads: [u8; MAX_SNAKES],
    pub necks: [u8; MAX_SNAKES],
    pub tails: [u8; MAX_SNAKES],
    pub health: [u8; MAX_SNAKES],
    pub length: [u8; MAX_SNAKES],
    pub alive_mask: u8,
    pub snake_count: u8,
    pub food_count: u8,
    pub food: [u8; MAX_FOOD],
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub const fn new() -> Self {
        GameState {
            grid: [EMPTY; BOARD_CELLS],
            heads: [SENTINEL; MAX_SNAKES],
            necks: [SENTINEL; MAX_SNAKES],
            tails: [SENTINEL; MAX_SNAKES],
            health: [0; MAX_SNAKES],
            length: [0; MAX_SNAKES],
            alive_mask: 0,
            snake_count: 0,
            food_count: 0,
            food: [SENTINEL; MAX_FOOD],
        }
    }

    #[inline(always)]
    pub const fn is_alive(&self, snake_id: usize) -> bool {
        (self.alive_mask & (1 << snake_id)) != 0
    }

    #[inline(always)]
    pub fn kill_snake(&mut self, snake_id: usize) {
        self.alive_mask &= !(1 << snake_id);
        self.health[snake_id] = 0;
    }

    #[inline(always)]
    pub const fn alive_count(&self) -> u32 {
        self.alive_mask.count_ones()
    }

    #[inline(always)]
    pub const fn is_game_over(&self) -> bool {
        self.alive_mask.count_ones() <= 1
    }

    #[inline(always)]
    pub fn is_cell_tail(&self, pos: u8) -> bool {
        for i in 0..self.snake_count as usize {
            if self.is_alive(i) && self.tails[i] == pos {
                return true;
            }
        }
        false
    }

    /// Get safe moves - excludes squares where larger/equal snakes can contest
    pub fn get_safe_moves(&self, snake_id: usize) -> ArrayVec<Direction, 4> {
        let mut moves = ArrayVec::new();

        if !self.is_alive(snake_id) {
            return moves;
        }

        let head = self.heads[snake_id];
        let neck = self.necks[snake_id];
        let my_length = self.length[snake_id];

        'outer: for dir in Direction::all() {
            if let Some(new_pos) = move_pos(head, dir) {
                if new_pos == neck {
                    continue;
                }

                // Check if any equal-or-larger enemy can also reach this cell
                for i in 0..self.snake_count as usize {
                    if i == snake_id || !self.is_alive(i) {
                        continue;
                    }

                    // Only worry about snakes that would kill us in head-to-head
                    if self.length[i] >= my_length {
                        let enemy_head = self.heads[i];
                        // Check all 4 directions from enemy head
                        for enemy_dir in Direction::all() {
                            if move_pos(enemy_head, enemy_dir) == Some(new_pos) {
                                continue 'outer; // Contested! Skip this move
                            }
                        }
                    }
                }

                moves.push(dir);
            }
        }

        // If ALL moves are contested, fall back to regular moves (take the gamble)
        if moves.is_empty() {
            return self.get_all_moves(snake_id);
        }

        moves
    }

    /// Get all possible moves (excluding neck reversal - guaranteed death)
    pub fn get_all_moves(&self, snake_id: usize) -> ArrayVec<Direction, 4> {
        let mut moves = ArrayVec::new();

        if !self.is_alive(snake_id) {
            return moves;
        }

        let head = self.heads[snake_id];
        let neck = self.necks[snake_id];

        for dir in Direction::all() {
            if let Some(new_pos) = move_pos(head, dir) {
                if new_pos != neck {
                    moves.push(dir);
                }
            }
        }

        if moves.is_empty() {
            for dir in Direction::all() {
                if move_pos(head, dir).is_some() {
                    moves.push(dir);
                    break;
                }
            }
        }

        moves
    }
}

// =============================================================================
// Turn Moves
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct TurnMoves {
    pub moves: [Option<Direction>; MAX_SNAKES],
}

impl TurnMoves {
    pub const fn new() -> Self {
        TurnMoves {
            moves: [None; MAX_SNAKES],
        }
    }

    pub fn set(&mut self, snake_id: usize, dir: Direction) {
        self.moves[snake_id] = Some(dir);
    }
}

// =============================================================================
// Move Application
// =============================================================================

impl GameState {
    pub fn apply_moves(&mut self, turn_moves: &TurnMoves) {
        let mut new_heads: [u8; MAX_SNAKES] = [SENTINEL; MAX_SNAKES];
        let mut ate_food: [bool; MAX_SNAKES] = [false; MAX_SNAKES];
        let mut eliminations: [bool; MAX_SNAKES] = [false; MAX_SNAKES];

        // Phase 1: Calculate new head positions
        for i in 0..self.snake_count as usize {
            if !self.is_alive(i) {
                continue;
            }

            let dir = match turn_moves.moves[i] {
                Some(d) => d,
                None => {
                    eliminations[i] = true;
                    continue;
                }
            };

            match move_pos(self.heads[i], dir) {
                Some(new_pos) => new_heads[i] = new_pos,
                None => eliminations[i] = true,
            }
        }

        // Phase 2: Check body collisions
        for i in 0..self.snake_count as usize {
            if eliminations[i] || !self.is_alive(i) {
                continue;
            }

            let new_head = new_heads[i];
            if new_head >= BOARD_CELLS as u8 {
                eliminations[i] = true;
                continue;
            }
            let cell = self.grid[new_head as usize];

            if cell != EMPTY && cell != FOOD {
                let mut is_moving_tail = false;
                for j in 0..self.snake_count as usize {
                    if self.is_alive(j) && self.tails[j] == new_head {
                        if new_heads[j] != SENTINEL && self.grid[new_heads[j] as usize] == FOOD {
                            is_moving_tail = false;
                        } else {
                            is_moving_tail = true;
                        }
                        break;
                    }
                }
                if !is_moving_tail {
                    eliminations[i] = true;
                }
            }
        }

        // Phase 3: Head-to-head collisions
        for i in 0..self.snake_count as usize {
            if eliminations[i] || !self.is_alive(i) {
                continue;
            }

            for j in (i + 1)..self.snake_count as usize {
                if eliminations[j] || !self.is_alive(j) {
                    continue;
                }

                if new_heads[i] == new_heads[j] {
                    let len_i = self.length[i];
                    let len_j = self.length[j];

                    if len_i > len_j {
                        eliminations[j] = true;
                    } else if len_j > len_i {
                        eliminations[i] = true;
                    } else {
                        eliminations[i] = true;
                        eliminations[j] = true;
                    }
                }
            }
        }

        // Phase 4: Check food consumption
        for i in 0..self.snake_count as usize {
            if eliminations[i] || !self.is_alive(i) {
                continue;
            }
            if self.grid[new_heads[i] as usize] == FOOD {
                ate_food[i] = true;
            }
        }

        // Phase 5: Move tails
        for i in 0..self.snake_count as usize {
            if !self.is_alive(i) || eliminations[i] {
                continue;
            }

            if !ate_food[i] {
                let old_tail = self.tails[i];
                if old_tail >= BOARD_CELLS as u8 {
                    eliminations[i] = true;
                    continue;
                }
                let new_tail = self.grid[old_tail as usize];
                self.grid[old_tail as usize] = EMPTY;
                self.tails[i] = new_tail;
            }
        }

        // Phase 6: Move heads
        for i in 0..self.snake_count as usize {
            if !self.is_alive(i) || eliminations[i] {
                continue;
            }

            let old_head = self.heads[i];
            let new_head = new_heads[i];
            if old_head >= BOARD_CELLS as u8 || new_head >= BOARD_CELLS as u8 {
                eliminations[i] = true;
                continue;
            }

            self.grid[old_head as usize] = new_head;
            self.grid[new_head as usize] = SENTINEL;
            self.necks[i] = old_head;
            self.heads[i] = new_head;

            if ate_food[i] {
                self.health[i] = MAX_HEALTH;
                self.length[i] += 1;
            } else {
                self.health[i] = self.health[i].saturating_sub(1);
                if self.health[i] == 0 {
                    eliminations[i] = true;
                }
            }
        }

        // Phase 7: Remove eaten food
        for i in 0..self.snake_count as usize {
            if ate_food[i] && self.is_alive(i) && !eliminations[i] {
                let food_pos = new_heads[i];
                for f in 0..self.food_count as usize {
                    if self.food[f] == food_pos {
                        self.food[f] = self.food[self.food_count as usize - 1];
                        self.food[self.food_count as usize - 1] = SENTINEL;
                        self.food_count -= 1;
                        break;
                    }
                }
            }
        }

        // Phase 8: Apply eliminations
        for i in 0..self.snake_count as usize {
            if eliminations[i] && self.is_alive(i) {
                let mut pos = self.heads[i];
                while pos != SENTINEL && pos < BOARD_CELLS as u8 {
                    let next = self.grid[pos as usize];
                    self.grid[pos as usize] = EMPTY;
                    if next == SENTINEL || next == pos {
                        break;
                    }
                    pos = next;
                }
                if self.tails[i] < BOARD_CELLS as u8 {
                    self.grid[self.tails[i] as usize] = EMPTY;
                }
                self.kill_snake(i);
            }
        }
    }
}

// =============================================================================
// API Types (Battlesnake server JSON format)
// =============================================================================

#[derive(Deserialize, Debug, Clone)]
pub struct ApiPosition {
    pub x: usize,
    pub y: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ApiSnake {
    pub id: String,
    pub health: u32,
    pub body: Vec<ApiPosition>,
    #[serde(default)]
    pub length: usize,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ApiRulesetSettings {
    #[serde(rename = "foodSpawnChance", default)]
    pub food_spawn_chance: u32,
    #[serde(rename = "minimumFood", default)]
    pub minimum_food: u32,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ApiRuleset {
    #[serde(default)]
    pub settings: Option<ApiRulesetSettings>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ApiGame {
    #[serde(default)]
    pub ruleset: Option<ApiRuleset>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ApiBoard {
    pub height: usize,
    pub width: usize,
    pub food: Vec<ApiPosition>,
    #[serde(default)]
    pub hazards: Vec<ApiPosition>,
    pub snakes: Vec<ApiSnake>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GameMoveRequest {
    pub game: ApiGame,
    pub turn: u32,
    pub board: ApiBoard,
    pub you: ApiSnake,
}

// =============================================================================
// Conversion from API types
// =============================================================================

impl GameState {
    pub fn from_api(request: &GameMoveRequest, my_snake_id: &str) -> (Self, usize) {
        let mut state = GameState::new();

        let mut my_index = 0;
        for (i, snake) in request.board.snakes.iter().enumerate() {
            if snake.id == my_snake_id {
                my_index = i;
                break;
            }
        }

        state.snake_count = request.board.snakes.len().min(MAX_SNAKES) as u8;

        let mut snake_order: Vec<usize> = (0..request.board.snakes.len()).collect();
        snake_order.swap(0, my_index);

        for (state_idx, &api_idx) in snake_order.iter().enumerate().take(MAX_SNAKES) {
            let snake = &request.board.snakes[api_idx];

            if snake.body.is_empty() {
                continue;
            }

            state.alive_mask |= 1 << state_idx;
            state.health[state_idx] = snake.health.min(255) as u8;
            state.length[state_idx] = snake.body.len() as u8;

            let head_pos = pos_to_idx(snake.body[0].x as u8, snake.body[0].y as u8);
            state.heads[state_idx] = head_pos;

            // Neck (second body segment)
            if snake.body.len() > 1 {
                state.necks[state_idx] = pos_to_idx(snake.body[1].x as u8, snake.body[1].y as u8);
            } else {
                state.necks[state_idx] = head_pos;
            }

            // Build linked list (tail -> head)
            for i in (1..snake.body.len()).rev() {
                let pos = pos_to_idx(snake.body[i].x as u8, snake.body[i].y as u8);
                let prev_pos = pos_to_idx(snake.body[i - 1].x as u8, snake.body[i - 1].y as u8);
                state.grid[pos as usize] = prev_pos;
            }

            let tail_pos = pos_to_idx(
                snake.body.last().unwrap().x as u8,
                snake.body.last().unwrap().y as u8,
            );
            state.tails[state_idx] = tail_pos;
            state.grid[head_pos as usize] = SENTINEL;
        }

        // Add food
        for food in &request.board.food {
            if state.food_count as usize >= MAX_FOOD {
                break;
            }
            let pos = pos_to_idx(food.x as u8, food.y as u8);
            state.grid[pos as usize] = FOOD;
            state.food[state.food_count as usize] = pos;
            state.food_count += 1;
        }

        (state, 0)
    }
}

// =============================================================================
// Move Enumeration
// =============================================================================

pub struct MoveEnumerator {
    state: GameState,
    snake_moves: [ArrayVec<Direction, 4>; MAX_SNAKES],
    indices: [usize; MAX_SNAKES],
    done: bool,
}

impl MoveEnumerator {
    pub fn new(state: &GameState) -> Self {
        let mut snake_moves = [
            ArrayVec::new(),
            ArrayVec::new(),
            ArrayVec::new(),
            ArrayVec::new(),
        ];

        for i in 0..state.snake_count as usize {
            if state.is_alive(i) {
                snake_moves[i] = state.get_all_moves(i);
            }
        }

        MoveEnumerator {
            state: *state,
            snake_moves,
            indices: [0; MAX_SNAKES],
            done: false,
        }
    }
}

impl Iterator for MoveEnumerator {
    type Item = TurnMoves;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut moves = TurnMoves::new();
        for i in 0..self.state.snake_count as usize {
            if self.state.is_alive(i) && !self.snake_moves[i].is_empty() {
                moves.moves[i] = Some(self.snake_moves[i][self.indices[i]]);
            }
        }

        let mut carry = true;
        for i in 0..self.state.snake_count as usize {
            if !self.state.is_alive(i) || self.snake_moves[i].is_empty() {
                continue;
            }

            if carry {
                self.indices[i] += 1;
                if self.indices[i] >= self.snake_moves[i].len() {
                    self.indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            self.done = true;
        }
        Some(moves)
    }
}

// =============================================================================
// Search Result (your main output)
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct MoveScore {
    pub direction: Direction,
    pub score: f64,      // The one metric: avg length diff
    pub leaf_count: u64, // How many futures we explored
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// The best move to make
    pub best_move: Direction,
    /// Score for each possible initial move
    pub move_scores: Vec<MoveScore>,
    /// Total tree nodes explored
    pub total_nodes: u64,
    /// Maximum depth reached in search
    pub max_depth: u32,
    /// Actual time spent searching
    pub elapsed_ms: u64,
    /// Throughput in millions of nodes per second
    pub throughput_mnps: f64,
}

// =============================================================================
// Main Search Function
// =============================================================================

/// Perform multiverse search on a Battlesnake game state.
///
/// This is the main entry point. Pass in the API request JSON and time budget,
/// get back the best move and statistics.
///
/// # Example
/// ```
/// let result = search(&request, Duration::from_millis(450));
/// println!("Move: {}", result.best_move);
/// ```
pub fn search(request: &GameMoveRequest, time_budget: Duration) -> SearchResult {
    let (state, my_snake) = GameState::from_api(request, &request.you.id);
    multiverse_search(&state, my_snake, time_budget)
}

pub fn multiverse_search(
    state: &GameState,
    my_snake: usize,
    time_budget: Duration,
) -> SearchResult {
    let start = Instant::now();
    let deadline = start + time_budget;

    let my_moves = state.get_safe_moves(my_snake);

    if my_moves.is_empty() {
        return SearchResult {
            best_move: Direction::Up,
            move_scores: vec![],
            total_nodes: 0,
            max_depth: 0,
            elapsed_ms: 0,
            throughput_mnps: 0.0,
        };
    }

    let mut length_diff_sum: [i64; 4] = [0; 4];
    let mut leaf_count: [u64; 4] = [0; 4];
    let mut total_nodes: u64 = 0;
    let mut max_depth: u32 = 0;

    let mut current_depth = 1u32;

    'outer: loop {
        for &my_dir in my_moves.iter() {
            for enemy_moves in EnemyMoveEnumerator::new(state, my_snake) {
                let mut turn_moves = TurnMoves::new();
                turn_moves.set(my_snake, my_dir);

                let mut enemy_idx = 0;
                for i in 0..state.snake_count as usize {
                    if i != my_snake && state.is_alive(i) {
                        if enemy_idx < enemy_moves.len() {
                            turn_moves.set(i, enemy_moves[enemy_idx]);
                            enemy_idx += 1;
                        }
                    }
                }

                let mut next_state = *state;
                next_state.apply_moves(&turn_moves);

                if total_nodes % 10000 == 0 && Instant::now() >= deadline {
                    break 'outer;
                }

                let (l, lc, branch_depth) = explore_branch(
                    &next_state,
                    my_snake,
                    current_depth - 1,
                    deadline,
                    &mut total_nodes,
                );

                length_diff_sum[my_dir as usize] += l;
                leaf_count[my_dir as usize] += lc;
                max_depth = max_depth.max(branch_depth + 1);

                if Instant::now() >= deadline {
                    break 'outer;
                }
            }
        }

        current_depth += 1;
        if current_depth > 100 {
            break;
        }
    }

    // Find best move - just highest average length diff!
    let mut best_dir = my_moves[0];
    let mut best_score = i64::MIN;

    for &dir in my_moves.iter() {
        let lc = leaf_count[dir as usize];
        let score = if lc > 0 {
            length_diff_sum[dir as usize] / lc as i64
        } else {
            i64::MIN
        };

        if score > best_score {
            best_score = score;
            best_dir = dir;
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis() as u64;
    let throughput = if elapsed.as_secs_f64() > 0.0 {
        total_nodes as f64 / elapsed.as_secs_f64() / 1_000_000.0
    } else {
        0.0
    };

    let move_scores: Vec<MoveScore> = my_moves
        .iter()
        .map(|&dir| {
            let lc = leaf_count[dir as usize];
            MoveScore {
                direction: dir,
                score: if lc > 0 {
                    length_diff_sum[dir as usize] as f64 / lc as f64
                } else {
                    f64::NEG_INFINITY
                },
                leaf_count: lc,
            }
        })
        .collect();

    SearchResult {
        best_move: best_dir,
        move_scores,
        total_nodes,
        max_depth,
        elapsed_ms,
        throughput_mnps: throughput,
    }
}

fn explore_branch(
    state: &GameState,
    my_snake: usize,
    depth: u32,
    deadline: Instant,
    total_nodes: &mut u64,
) -> (i64, u64, u32) {
    // (length_diff_sum, leaf_count, max_depth)
    *total_nodes += 1;

    if !state.is_alive(my_snake) {
        // Dead = catastrophically negative
        return (-50, 1, 0);
    }

    if state.alive_count() == 1 {
        // Won! Our length is pure advantage
        return (state.length[my_snake] as i64, 1, 0);
    }

    if depth == 0 || Instant::now() >= deadline {
        // Ongoing: how much longer are we than the biggest threat?
        let my_len = state.length[my_snake] as i64;
        let max_enemy_len = (0..state.snake_count as usize)
            .filter(|&i| i != my_snake && state.is_alive(i))
            .map(|i| state.length[i] as i64)
            .max()
            .unwrap_or(0);

        return (my_len - max_enemy_len, 1, 0);
    }

    let mut length_diff_sum = 0i64;
    let mut leaf_count = 0u64;
    let mut max_depth = 0u32;

    for moves in MoveEnumerator::new(state) {
        let mut next_state = *state;
        next_state.apply_moves(&moves);

        let (l, lc, d) = explore_branch(&next_state, my_snake, depth - 1, deadline, total_nodes);
        length_diff_sum += l;
        leaf_count += lc;
        max_depth = max_depth.max(d + 1);

        if *total_nodes % 50000 == 0 && Instant::now() >= deadline {
            break;
        }
    }

    (length_diff_sum, leaf_count, max_depth)
}

struct EnemyMoveEnumerator {
    moves: Vec<ArrayVec<Direction, 4>>,
    indices: Vec<usize>,
    done: bool,
}

impl EnemyMoveEnumerator {
    fn new(state: &GameState, my_snake: usize) -> Self {
        let mut moves = Vec::new();

        for i in 0..state.snake_count as usize {
            if i != my_snake && state.is_alive(i) {
                let snake_moves = state.get_all_moves(i);
                if !snake_moves.is_empty() {
                    moves.push(snake_moves);
                }
            }
        }

        let indices = vec![0; moves.len()];
        let done = moves.is_empty();

        EnemyMoveEnumerator {
            moves,
            indices,
            done,
        }
    }
}

impl Iterator for EnemyMoveEnumerator {
    type Item = Vec<Direction>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result: Vec<Direction> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| self.moves[i][idx])
            .collect();

        let mut carry = true;
        for i in 0..self.indices.len() {
            if carry {
                self.indices[i] += 1;
                if self.indices[i] >= self.moves[i].len() {
                    self.indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            self.done = true;
        }
        Some(result)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_api() {
        let json = r#"{
            "game": {"ruleset": {"settings": {}}},
            "turn": 10,
            "board": {
                "height": 11, "width": 11,
                "food": [{"x": 5, "y": 5}],
                "hazards": [],
                "snakes": [
                    {"id": "me", "health": 95, "body": [{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}], "length": 3},
                    {"id": "enemy", "health": 90, "body": [{"x": 8, "y": 8}, {"x": 8, "y": 9}, {"x": 8, "y": 10}], "length": 3}
                ]
            },
            "you": {"id": "me", "health": 95, "body": [{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}], "length": 3}
        }"#;

        let request: GameMoveRequest = serde_json::from_str(json).unwrap();
        let result = search(&request, Duration::from_millis(50));

        assert!(!result.move_scores.is_empty());
        assert!(result.total_nodes > 0);
        println!("Best move: {}", result.best_move);
        println!("Nodes: {}, Depth: {}", result.total_nodes, result.max_depth);
    }

    #[test]
    fn test_game_state_size() {
        let size = std::mem::size_of::<GameState>();
        println!("GameState size: {} bytes", size);
        assert!(size <= 170);
    }
}
