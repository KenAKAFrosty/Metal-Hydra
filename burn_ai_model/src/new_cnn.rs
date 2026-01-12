use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Configuration for the BattleSnake CNN model.
///
/// Architecture designed for 11x11 board with 8 input channels:
/// - 5 conv layers (3x3) for full receptive field coverage
/// - BatchNorm after each conv
/// - ReLU activations
/// - Optional spatial dropout after conv blocks
/// - MaxPool at the end to compress spatial dimensions  
/// - MLP head with metadata concatenation and optional dropout
#[derive(Config, Debug)]
pub struct BattleCnnConfig {
    /// Number of input channels (board encoding)
    #[config(default = 8)]
    pub in_channels: usize,

    /// Number of metadata features
    #[config(default = 3)]
    pub meta_features: usize,

    /// Number of output classes (directions: up, right, down, left)
    #[config(default = 4)]
    pub num_classes: usize,

    /// Grid size (assumes square board)
    #[config(default = 11)]
    pub grid_size: usize,

    /// Dropout rate for MLP head. 0.0 = disabled.
    #[config(default = 0.0)]
    pub mlp_dropout: f64,
}

#[derive(Module, Debug)]
pub struct BattleCnn<B: Backend> {
    // Convolutional layers - 5 layers for full 11x11 receptive field
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,

    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,

    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,

    conv4: Conv2d<B>,
    bn4: BatchNorm<B>,

    conv5: Conv2d<B>,
    bn5: BatchNorm<B>,

    // Pooling
    maxpool: MaxPool2d,

    // MLP Head
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,

    // MLP dropout
    mlp_dropout: Dropout,
}

impl<B: Backend> BattleCnn<B> {
    pub fn new(config: &BattleCnnConfig, device: &B::Device) -> Self {
        // Conv layer helper - 3x3, stride 1, padding 1 (preserves spatial dims)
        let conv3x3 = |in_ch: usize, out_ch: usize| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device)
        };

        // Layer 1: 8 -> 32 channels, RF: 3x3
        let conv1 = conv3x3(config.in_channels, 32);
        let bn1 = BatchNormConfig::new(32).init(device);

        // Layer 2: 32 -> 32 channels, RF: 5x5
        let conv2 = conv3x3(32, 32);
        let bn2 = BatchNormConfig::new(32).init(device);

        // Layer 3: 32 -> 64 channels, RF: 7x7
        let conv3 = conv3x3(32, 64);
        let bn3 = BatchNormConfig::new(64).init(device);

        // Layer 4: 64 -> 64 channels, RF: 9x9
        let conv4 = conv3x3(64, 64);
        let bn4 = BatchNormConfig::new(64).init(device);

        // Layer 5: 64 -> 64 channels, RF: 11x11 ✓ Full board coverage
        let conv5 = conv3x3(64, 64);
        let bn5 = BatchNormConfig::new(64).init(device);

        // MaxPool: 11x11 -> 5x5
        let maxpool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();

        // After pooling: 64 channels * 5 * 5 = 1600
        // Plus metadata: 1600 + meta_features
        let flattened_size = 64 * 5 * 5;
        let fc1_input = flattened_size + config.meta_features;

        // MLP Head
        let fc1 = LinearConfig::new(fc1_input, 256)
            .with_bias(true)
            .init(device);
        let fc2 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let fc3 = LinearConfig::new(128, config.num_classes)
            .with_bias(true)
            .init(device);

        // MLP dropout
        let mlp_dropout = DropoutConfig::new(config.mlp_dropout).init();

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
            bn4,
            conv5,
            bn5,
            maxpool,
            fc1,
            fc2,
            fc3,
            mlp_dropout,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `board` - Board tensor of shape [B, 8, 11, 11]
    /// * `metadata` - Metadata tensor of shape [B, 3]
    ///
    /// # Returns
    /// Logits tensor of shape [B, 4] (one per direction)
    pub fn forward(&self, board: Tensor<B, 4>, metadata: Tensor<B, 2>) -> Tensor<B, 2> {
        // Conv block 1
        let x = self.conv1.forward(board);
        let x = self.bn1.forward(x);
        let x = relu(x);

        // Conv block 2
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = relu(x);

        // Conv block 3
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = relu(x);

        // Conv block 4
        let x = self.conv4.forward(x);
        let x = self.bn4.forward(x);
        let x = relu(x);

        // Conv block 5 - Now every spatial location has seen the full board
        let x = self.conv5.forward(x);
        let x = self.bn5.forward(x);
        let x = relu(x);

        // Pooling: [B, 64, 11, 11] -> [B, 64, 5, 5]
        let x = self.maxpool.forward(x);

        // Flatten: [B, 64, 5, 5] -> [B, 1600]
        let batch_size = x.dims()[0];
        let x = x.reshape([batch_size as i32, -1]);

        // Concatenate metadata: [B, 1600] + [B, 3] -> [B, 1603]
        let x = Tensor::cat(vec![x, metadata], 1);

        // MLP Head with dropout between layers
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.mlp_dropout.forward(x);

        let x = self.fc2.forward(x);
        let x = relu(x);
        let x = self.mlp_dropout.forward(x);

        // Output logits (no activation - CrossEntropy expects raw logits)
        self.fc3.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let config = BattleCnnConfig::new();
        let model: BattleCnn<TestBackend> = BattleCnn::new(&config, &device);

        // Create dummy inputs
        let batch_size = 4;
        let board = Tensor::zeros([batch_size, 8, 11, 11], &device);
        let metadata = Tensor::zeros([batch_size, 3], &device);

        let output = model.forward(board, metadata);

        assert_eq!(output.dims(), [batch_size, 4]);
    }

    #[test]
    fn test_with_dropout() {
        let device = Default::default();
        let config = BattleCnnConfig {
            in_channels: 8,
            meta_features: 3,
            num_classes: 4,
            grid_size: 11,
            mlp_dropout: 0.15,
        };
        let model: BattleCnn<TestBackend> = BattleCnn::new(&config, &device);

        let board = Tensor::zeros([2, 8, 11, 11], &device);
        let metadata = Tensor::zeros([2, 3], &device);

        let output = model.forward(board, metadata);

        assert_eq!(output.dims(), [2, 4]);
    }
}
