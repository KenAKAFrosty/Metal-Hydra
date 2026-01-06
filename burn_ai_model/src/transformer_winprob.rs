use burn::prelude::*;
use burn::{
    module::Param,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Dropout, DropoutConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::{gelu, softmax, tanh},
        Distribution,
    },
};

#[derive(Config, Debug)]
pub struct BattleModelConfig {
    // --- Transformer Architecture ---
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 2)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,

    // --- Data Dimensions ---
    pub num_classes: usize,
    #[config(default = 27)] // Updated to include is_empty
    pub tile_features: usize,
    pub meta_features: usize,
    pub grid_size: usize,

    // --- Head Architecture (New) ---
    #[config(default = 8)]
    pub num_queries: usize, // Number of "detectives" scanning the board
    #[config(default = 128)]
    pub head_hidden_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct BattleModel<B: Backend> {
    tile_projection: Linear<B>,
    pos_projection: Linear<B>,
    meta_projection: Linear<B>,
    transformer: TransformerEncoder<B>,

    // --- NEW: Pooling ---
    // Learnable query vectors [1, num_queries, d_model]
    pooling_queries: Param<Tensor<B, 3>>,

    // --- NEW: MLP Head ---
    head_dense: Linear<B>,
    dropout: Dropout,
    output: Linear<B>,

    // --- OPTIMIZATION: Grid ---
    pos_grid: Param<Tensor<B, 3>>,
}

impl<B: Backend> BattleModel<B> {
    pub fn new(config: &BattleModelConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;

        // 1. Projections
        // Note: tile_features should now be 27 (26 original + 1 is_empty)
        let tile_projection = LinearConfig::new(config.tile_features, d_model).init(device);
        let pos_projection = LinearConfig::new(2, d_model).init(device);
        let meta_projection = LinearConfig::new(config.meta_features, d_model).init(device);

        // 2. Transformer (Leaner config)
        let transformer =
            TransformerEncoderConfig::new(d_model, config.d_ff, config.n_heads, config.n_layers)
                .init(device);

        // 3. Pooling Queries
        // We initialize these randomly. They will learn to look for specific patterns.
        // Shape: [1, num_queries, d_model]
        let queries = Tensor::random(
            [1, config.num_queries, d_model],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        let pooling_queries = Param::from_tensor(queries);

        // 4. MLP Head
        // Input size = (Number of Queries * d_model) + (Meta Embed Size)
        let pooled_size = config.num_queries * d_model;
        let head_input_size = pooled_size + d_model; // meta_projection outputs d_model

        let head_dense = LinearConfig::new(head_input_size, config.head_hidden_size).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let output = LinearConfig::new(config.head_hidden_size, config.num_classes).init(device);

        // 5. Grid Optimization (Same as before)
        let size = config.grid_size;
        let mut coords = Vec::with_capacity(size * size * 2);
        for y in 0..size {
            for x in 0..size {
                coords.push(x as f32 / (size - 1) as f32);
                coords.push(y as f32 / (size - 1) as f32);
            }
        }
        let grid_data = burn::tensor::TensorData::new(coords, vec![1, size * size, 2]);
        let grid_tensor = Tensor::<B, 3>::from_floats(grid_data, device);
        let pos_grid = Param::from_tensor(grid_tensor).set_require_grad(false);

        Self {
            tile_projection,
            pos_projection,
            meta_projection,
            transformer,
            pooling_queries, // New
            head_dense,      // New
            dropout,         // New
            output,
            pos_grid,
        }
    }

    pub fn forward(&self, tiles: Tensor<B, 3>, metadata: Tensor<B, 2>) -> Tensor<B, 2> {
        // tiles shape: [Batch, 121, 27] (assuming is_empty added)

        // 1. Embeddings
        let x_tiles = self.tile_projection.forward(tiles);
        let pos_embeds = self.pos_projection.forward(self.pos_grid.val());
        let x = x_tiles + pos_embeds; // Broadcasting applies here

        // 2. Transformer [Batch, 121, d_model]
        let encoded = self.transformer.forward(TransformerEncoderInput::new(x));

        // 3. Multi-Query Attention Pooling
        // Q: [1, K, D]
        let q = self.pooling_queries.val();

        // K_T: [Batch, D, 121] (Transpose of encoded)
        let k_t = encoded.clone().transpose();

        // Scores: [Batch, K, 121]
        // Q broadcasts to [Batch, K, D] automatically
        let d_scale = (encoded.dims()[2] as f32).sqrt();
        let att_scores = q.matmul(k_t) / d_scale;

        // Weights: Softmax over the board tiles (dim 2)
        // Each query decides which tiles are important
        let att_weights = softmax(att_scores, 2);

        // Pooled: [Batch, K, 121] @ [Batch, 121, D] -> [Batch, K, D]
        let pooled_vectors = att_weights.matmul(encoded);

        // Flatten queries: [Batch, K * D]
        let pooled_flat = pooled_vectors.flatten(1, 2);

        // 4. Metadata
        let meta_embed = self.meta_projection.forward(metadata);

        // 5. Concatenate
        // [Batch, (K*D) + D]
        let mlp_input = Tensor::cat(vec![pooled_flat, meta_embed], 1);

        // 6. MLP Head
        let x = self.head_dense.forward(mlp_input);
        let x = gelu(x); // Non-linearity allows complex logic
        let x = self.dropout.forward(x);

        // CHANGED FROM ORIGINAL TRANSFORMER: From Classification to Regression
        let logits = self.output.forward(x);

        // We apply Tanh to squash the output between -1.0 and 1.0.
        // This matches our target values (Outcome * Decay).
        // 1.0 = Guaranteed Win, -1.0 = Guaranteed Loss.
        tanh(logits)
    }
}
