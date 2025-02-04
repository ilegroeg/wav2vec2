use core::fmt::Debug;

use burn::module::Module;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Gelu;
use burn::nn::GroupNorm;
use burn::nn::GroupNormConfig;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::prelude::Backend;
use burn::tensor::Device;
use burn::tensor::Tensor;

use crate::components::config::Wav2Vec2Config;

#[derive(Module, Debug)]
pub struct Wav2Vec2FeatureEncoder<B: Backend> {
    pub conv_layers: Vec<Wav2Vec2ConvLayer<B>>,
}

impl<B: Backend> Wav2Vec2FeatureEncoder<B> {
    pub fn init(config: &Wav2Vec2Config, device: &Device<B>) -> Self {
        let mut conv_layers = Vec::new();

        match config.feat_extract_norm.as_str() {
            "layer" => {
                for i in 0..config.num_feat_extract_layers {
                    conv_layers.push(Wav2Vec2ConvLayer::LayerNorm(Wav2Vec2LayerNormConvLayer::init(config, i, device)));
                }
            }
            "group" => {
                conv_layers.push(Wav2Vec2ConvLayer::GroupNorm(Wav2Vec2GroupNormConvLayer::init(config, 0, device)));
                for i in 1..config.num_feat_extract_layers {
                    conv_layers.push(Wav2Vec2ConvLayer::NoLayerNorm(Wav2Vec2NoLayerNormConvLayer::init(config, i, device)));
                }
            }
            _ => {
                panic!("Invalid feat_extract_norm: {}", config.feat_extract_norm);
            }
        }

        Self { conv_layers }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let mut hidden_states = input.unsqueeze_dim(1);
        for conv_layer in &self.conv_layers {
            hidden_states = conv_layer.forward(hidden_states);
        }
        hidden_states
    }
}

#[derive(Module, Debug)]
pub enum Wav2Vec2ConvLayer<B: Backend> {
    LayerNorm(Wav2Vec2LayerNormConvLayer<B>),
    NoLayerNorm(Wav2Vec2NoLayerNormConvLayer<B>),
    GroupNorm(Wav2Vec2GroupNormConvLayer<B>),
}

impl<B: Backend> Wav2Vec2ConvLayer<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Wav2Vec2ConvLayer::LayerNorm(layer) => layer.forward(input),
            Wav2Vec2ConvLayer::NoLayerNorm(layer) => layer.forward(input),
            Wav2Vec2ConvLayer::GroupNorm(layer) => layer.forward(input),
        }
    }

    pub (crate) fn conv(&self) -> &Conv1d<B> {
        match self {
            Wav2Vec2ConvLayer::LayerNorm(layer) => &layer.conv,
            Wav2Vec2ConvLayer::NoLayerNorm(layer) => &layer.conv,
            Wav2Vec2ConvLayer::GroupNorm(layer) => &layer.conv,
        }
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2LayerNormConvLayer<B: Backend> {
    pub conv: Conv1d<B>,
    layer_norm: LayerNorm<B>,
    activation: Gelu,
}

impl<B: Backend> Wav2Vec2LayerNormConvLayer<B> {
    pub fn init(config: &Wav2Vec2Config, layer_index: usize, device: &Device<B>) -> Self {
        // shape (batch_size, in_channels, original_seq_len) -> (batch_size, out_channels, new_seq_len)
        let in_channels = if layer_index == 0 {
            1
        } else {
            config.conv_dim[layer_index - 1]
        };
        let out_channels = config.conv_dim[layer_index];
        let conv = Conv1dConfig::new(in_channels, out_channels, config.conv_kernel[layer_index])
            .with_stride(config.conv_stride[layer_index])
            .with_bias(config.conv_bias)
            .init(device);
        let layer_norm = LayerNormConfig::new(out_channels)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let activation = Gelu::new();

        Self {
            conv,
            layer_norm,
            activation,
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.conv.forward(hidden_states);

        let hidden_states = hidden_states.swap_dims(1, 2); // shape (batch_size, out_channels, new_seq_len) -> (batch_size, new_seq_len, out_channels)
        let hidden_states = self.layer_norm.forward(hidden_states);
        let hidden_states = hidden_states.swap_dims(1, 2); // shape (batch_size, new_seq_len, out_channels) -> (batch_size, out_channels, new_seq_len)

        self.activation.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2NoLayerNormConvLayer<B: Backend> {
    conv: Conv1d<B>,
    activation: Gelu,
}

impl<B: Backend> Wav2Vec2NoLayerNormConvLayer<B> {
    fn init(config: &Wav2Vec2Config, layer_index: usize, device: &Device<B>) -> Self {
        // shape (batch_size, in_channels, original_seq_len) -> (batch_size, out_channels, new_seq_len)
        let in_channels = if layer_index == 0 {
            1
        } else {
            config.conv_dim[layer_index - 1]
        };
        let out_channels = config.conv_dim[layer_index];
        let conv = Conv1dConfig::new(in_channels, out_channels, config.conv_kernel[layer_index])
            .with_stride(config.conv_stride[layer_index])
            .with_bias(config.conv_bias)
            .init(device);
        let activation = Gelu::new();

        Self { conv, activation }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.conv.forward(hidden_states);
        self.activation.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2GroupNormConvLayer<B: Backend> {
    conv: Conv1d<B>,
    layer_norm: GroupNorm<B>,
    activation: Gelu,
}

impl<B: Backend> Wav2Vec2GroupNormConvLayer<B> {
    fn init(config: &Wav2Vec2Config, layer_index: usize, device: &Device<B>) -> Self {
        // shape (batch_size, in_channels, original_seq_len) -> (batch_size, out_channels, new_seq_len)
        let in_channels = if layer_index == 0 {
            1
        } else {
            config.conv_dim[layer_index - 1]
        };
        let out_channels = config.conv_dim[layer_index];
        let conv = Conv1dConfig::new(in_channels, out_channels, config.conv_kernel[layer_index])
            .with_stride(config.conv_stride[layer_index])
            .with_bias(config.conv_bias)
            .init(device);
        let layer_norm = GroupNormConfig::new(out_channels, out_channels)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let activation = Gelu::new();

        Self {
            conv,
            layer_norm,
            activation,
        }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.conv.forward(hidden_states);

        let hidden_states = hidden_states.swap_dims(1, 2); // shape (batch_size, out_channels, new_seq_len) -> (batch_size, new_seq_len, out_channels)
        let hidden_states = self.layer_norm.forward(hidden_states);
        let hidden_states = hidden_states.swap_dims(1, 2); // shape (batch_size, new_seq_len, out_channels) -> (batch_size, out_channels, new_seq_len)

        self.activation.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2FeatureProjection<B: Backend> {
    layer_norm: LayerNorm<B>,
    projection: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Wav2Vec2FeatureProjection<B> {
    pub fn init(in_features: usize, out_features: usize, dropout: f64, device: &Device<B>) -> Self {
        let layer_norm = LayerNormConfig::new(in_features).init(device);
        let projection = LinearConfig::new(in_features, out_features)
            .with_bias(true)
            .init(device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            layer_norm,
            projection,
            dropout,
        }
    }

    pub fn forward<const D: usize>(
        &self,
        hidden_states: Tensor<B, D>,
    ) -> (Tensor<B, D>, Tensor<B, D>) {
        let norm_hidden_states = self.layer_norm.forward(hidden_states.clone());
        let hidden_states = self.projection.forward(norm_hidden_states.clone());
        (self.dropout.forward(hidden_states), norm_hidden_states)
    }
}
