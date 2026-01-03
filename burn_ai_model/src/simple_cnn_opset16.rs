// Generated from ONNX "simple_cnn_opset16.onnx" by burn-import
use burn::prelude::*;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    gemm1: Linear<B>,
    gemm2: Linear<B>,
    gemm3: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("src/model/simple_cnn_opset16", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([8, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let gemm1 = LinearConfig::new(3208, 256).with_bias(true).init(device);
        let gemm2 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let gemm3 = LinearConfig::new(128, 4).with_bias(true).init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            gemm1,
            gemm2,
            gemm3,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>, input2: Tensor<B, 2>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu3_out1);
        let flatten1_out1 = {
            let leading_dim = maxpool2d1_out1.shape().dims[..1].iter().product::<usize>()
                as i32;
            maxpool2d1_out1.reshape::<2, _>([leading_dim, -1])
        };
        let concat1_out1 = burn::tensor::Tensor::cat([flatten1_out1, input2].into(), 1);
        let gemm1_out1 = self.gemm1.forward(concat1_out1);
        let relu4_out1 = burn::tensor::activation::relu(gemm1_out1);
        let gemm2_out1 = self.gemm2.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(gemm2_out1);
        let gemm3_out1 = self.gemm3.forward(relu5_out1);
        gemm3_out1
    }
}
