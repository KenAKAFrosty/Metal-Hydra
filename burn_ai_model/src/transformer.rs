use burn::prelude::*;
use burn::{
    module::Param,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    },
};

#[derive(Config, Debug)]
pub struct BattleModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub num_classes: usize,
    pub tile_features: usize,
    pub meta_features: usize,
    pub grid_size: usize,
}

#[derive(Module, Debug)]
pub struct BattleModel<B: Backend> {
    tile_projection: Linear<B>,
    pos_projection: Linear<B>,
    meta_projection: Linear<B>,
    transformer: TransformerEncoder<B>,
    output: Linear<B>,

    // CHANGED: We store the grid here.
    // Wrapped in Param so it moves devices with the model,
    // but we will turn off gradients.
    pos_grid: Param<Tensor<B, 3>>,
}

impl<B: Backend> BattleModel<B> {
    pub fn new(config: &BattleModelConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;

        let tile_projection = LinearConfig::new(config.tile_features, d_model).init(device);
        let pos_projection = LinearConfig::new(2, d_model).init(device);
        let meta_projection = LinearConfig::new(config.meta_features, d_model).init(device);

        let transformer =
            TransformerEncoderConfig::new(d_model, config.d_ff, config.n_heads, config.n_layers)
                .init(device);

        let output = LinearConfig::new(d_model * 2, config.num_classes).init(device);

        // --- OPTIMIZATION START ---
        // 1. Generate the grid ONCE on CPU
        let size = config.grid_size;
        let mut coords = Vec::with_capacity(size * size * 2);
        for y in 0..size {
            for x in 0..size {
                coords.push(x as f32 / (size - 1) as f32);
                coords.push(y as f32 / (size - 1) as f32);
            }
        }

        // 2. Create tensor on Device
        let grid_data = burn::tensor::TensorData::new(coords, vec![1, size * size, 2]);
        let grid_tensor = Tensor::<B, 3>::from_floats(grid_data, device);

        // 3. Wrap in Param and disable gradients.
        // This acts like a "Fixed Buffer" in other frameworks.
        let pos_grid = Param::from_tensor(grid_tensor).set_require_grad(false);
        // --- OPTIMIZATION END ---

        Self {
            tile_projection,
            pos_projection,
            meta_projection,
            transformer,
            output,
            pos_grid,
        }
    }

    pub fn forward(&self, tiles: Tensor<B, 3>, metadata: Tensor<B, 2>) -> Tensor<B, 2> {
        // tiles shape: [Batch, 121, Feat]

        // 1. Embed Tiles [Batch, 121, d_model]
        let x_tiles = self.tile_projection.forward(tiles);

        // 2. Embed Positions
        // self.pos_grid is [1, 121, 2].
        // Linear projection works on the last dim, resulting in [1, 121, d_model].
        // We use .val() to get the tensor from the Param.
        let pos_embeds = self.pos_projection.forward(self.pos_grid.val());

        // 3. Combine (Broadcasting)
        // [Batch, 121, d_model] + [1, 121, d_model]
        // Burn automatically expands the position embeddings to match the batch size.
        // This is significantly faster than repeat_dim.
        let x = x_tiles + pos_embeds;

        // 4. Run Transformer
        let encoded = self.transformer.forward(TransformerEncoderInput::new(x));

        // Note: mean_dim and squeeze are view/compute ops.
        // Ensure you are using the fused operations if available in your backend,
        // but this specific sequence is standard.
        let pooled = encoded.mean_dim(1).squeeze_dim(1);

        let meta_embed = self.meta_projection.forward(metadata);

        // Concatenation is necessary here, but ensures we don't hold references to
        // `pooled` or `meta_embed` longer than necessary to aid Fusion.
        let mlp_input = Tensor::cat(vec![pooled, meta_embed], 1);

        self.output.forward(mlp_input)
    }
}
