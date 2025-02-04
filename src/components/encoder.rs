use burn::module::Module;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Gelu;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig1d;
use burn::prelude::Backend;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;

use crate::components::config::Wav2Vec2Config;
use crate::components::util::ParameterizedConv1d;
use crate::components::util::ParameterizedConv1dConfig;

#[derive(Module, Debug)]
pub struct Wav2Vec2EncoderStableLayerNorm<B: Backend> {
    pub pos_conv_embed: Wav2Vec2PositionalConvEmbedding<B>,
    pub layer_norm: LayerNorm<B>,
    pub dropout: Dropout,
    pub layers: Vec<Wav2Vec2EncoderLayerStableLayerNorm<B>>,
}

impl<B: Backend> Wav2Vec2EncoderStableLayerNorm<B> {
    pub fn init(config: &Wav2Vec2Config, device: &Device<B>) -> Self {
        let pos_conv_embed = Wav2Vec2PositionalConvEmbedding::init(
            config.hidden_size,
            config.num_conv_pos_embeddings,
            config.num_conv_pos_embedding_groups,
            device,
        );
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.hidden_dropout).init();
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(Wav2Vec2EncoderLayerStableLayerNorm::init(config, device));
        }
        Self {
            pos_conv_embed,
            layer_norm,
            dropout,
            layers,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // don't attend to padded tokens
        let hidden_states = if let Some(mask) = &attn_mask {
            hidden_states.mul(mask.clone().unsqueeze_dim::<3>(2))
        } else {
            hidden_states
        };

        let attn_mask = attn_mask.map(|mask| {
            let mask_dims = mask.shape();
            // (batch_size, 1, 1, seqlen)
            let reshaped_mask = mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
            let negated_mask = Tensor::ones_like(&reshaped_mask).sub(reshaped_mask);
            // f32::MIN or -10000.0 depending on transformers version
            let large_neg_val = -10000.0;
            let lnn_mask = negated_mask.mul_scalar(large_neg_val);
            lnn_mask.expand([mask_dims.dims[0], 1, mask_dims.dims[1], mask_dims.dims[1]])
        });
        let hidden_states = hidden_states
            .clone()
            .add(self.pos_conv_embed.forward(hidden_states));

        let hidden_states = self.dropout.forward(hidden_states);

        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, attn_mask.clone());
        }

        self.layer_norm.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2PositionalConvEmbedding<B: Backend> {
    pub conv: ParameterizedConv1d<B>,
    pub padding: usize,
    pub activation: Gelu,
}

impl<B: Backend> Wav2Vec2PositionalConvEmbedding<B> {
    pub fn init(
        hidden_size: usize,
        num_conv_pos_embeddings: usize,
        num_conv_pos_embedding_groups: usize,
        device: &Device<B>,
    ) -> Self {
        let conv =
            ParameterizedConv1dConfig::new(hidden_size, hidden_size, num_conv_pos_embeddings)
                .with_weight_norm_dim(Some(2))
                .with_padding(PaddingConfig1d::Explicit(num_conv_pos_embeddings / 2))
                .with_groups(num_conv_pos_embedding_groups)
                .init(device);
        let padding = if num_conv_pos_embeddings % 2 == 0 {
            1
        } else {
            0
        };
        let activation = Gelu::new();
        Self {
            conv,
            padding,
            activation,
        }
    }

    fn fix_padding(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.padding > 0 {
            let dims = input.dims();
            input.slice([
                (0_i64, dims[0] as i64),
                (0_i64, dims[1] as i64),
                (0_i64, dims[2] as i64 - self.padding as i64),
            ])
        } else {
            input
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = hidden_states.swap_dims(1, 2);

        let hidden_states = self.conv.forward(hidden_states);
        let hidden_states = self.fix_padding(hidden_states);
        let hidden_states = self.activation.forward(hidden_states);

        hidden_states.swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2EncoderLayerStableLayerNorm<B: Backend> {
    pub attention: Wav2Vec2Attention<B>,
    pub dropout: Dropout,
    pub layer_norm: LayerNorm<B>,
    pub feed_forward: Wav2Vec2FeedForward<B>,
    pub final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> Wav2Vec2EncoderLayerStableLayerNorm<B> {
    pub fn init(config: &Wav2Vec2Config, device: &Device<B>) -> Self {
        let attention = Wav2Vec2Attention::init(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            true,
            device,
        );
        let dropout = DropoutConfig::new(config.hidden_dropout).init();
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let feed_forward = Wav2Vec2FeedForward::init(
            config.hidden_size,
            config.intermediate_size,
            true,
            config.activation_dropout,
            config.hidden_dropout,
            device,
        );
        let final_layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        Self {
            attention,
            dropout,
            layer_norm,
            feed_forward,
            final_layer_norm,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let attn_residual = hidden_states.clone();
        let hidden_states = self.layer_norm.forward(hidden_states);
        let hidden_states = self.attention.forward(hidden_states, attn_mask);
        let hidden_states = self.dropout.forward(hidden_states);
        let hidden_states = attn_residual.add(hidden_states);
        hidden_states.clone().add(
            self.feed_forward
                .forward(self.final_layer_norm.forward(hidden_states)),
        )
    }
}

// Multi-head attention from Attention Is All You Need: https://arxiv.org/abs/1706.03762
#[derive(Module, Debug)]
pub struct Wav2Vec2Attention<B: Backend> {
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub q_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub embed_dim: usize,
    pub dropout: f64,
    pub head_dim: usize,
    pub scaling: f32,
}

impl<B: Backend> Wav2Vec2Attention<B> {
    pub fn init(
        embed_dim: usize,
        num_heads: usize,
        dropout: f64,
        bias: bool,
        device: &Device<B>,
    ) -> Self {
        let head_dim = embed_dim / num_heads;

        if head_dim * num_heads != embed_dim {
            panic!("embed_dim must be divisible by num_heads");
        }

        let scaling = (head_dim as f32).powf(-0.5);
        let k_proj = LinearConfig::new(embed_dim, embed_dim)
            .with_bias(bias)
            .init(device);
        let v_proj = LinearConfig::new(embed_dim, embed_dim)
            .with_bias(bias)
            .init(device);
        let q_proj = LinearConfig::new(embed_dim, embed_dim)
            .with_bias(bias)
            .init(device);
        let out_proj = LinearConfig::new(embed_dim, embed_dim)
            .with_bias(bias)
            .init(device);
        Self {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            num_heads,
            embed_dim,
            dropout,
            head_dim,
            scaling,
        }
    }

    fn _shape(&self, tensor: Tensor<B, 3>, seq_len: i32, batch_size: i32) -> Tensor<B, 3> {
        tensor
            .reshape([
                batch_size,
                seq_len,
                self.num_heads as i32,
                self.head_dim as i32,
            ])
            .swap_dims(1, 2)
            .reshape([
                (batch_size * self.num_heads as i32),
                -1, // infer from other dims
                self.head_dim as i32,
            ])
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>, // (batch, time, channel)
        attn_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch_size, tgt_len, _] = hidden_states.dims();

        let query = self.q_proj.forward(hidden_states.clone()) * self.scaling;
        let key = self.k_proj.forward(hidden_states.clone());
        let value = self.v_proj.forward(hidden_states);

        let query_states = self._shape(query, -1, batch_size as i32);
        let key_states = self._shape(key, -1, batch_size as i32);
        let value_states = self._shape(value, -1, batch_size as i32);

        let src_len = key_states.dims()[1];
        let mut attn_weights = query_states.matmul(key_states.swap_dims(1, 2));

        if attn_weights.dims() != [batch_size * self.num_heads, tgt_len, src_len] {
            panic!(
                "attn_weights should be of shape {:?}, but is {:?}",
                [batch_size * self.num_heads, tgt_len, src_len],
                attn_weights.dims()
            );
        }

        if let Some(attn_mask) = attn_mask {
            if attn_mask.dims() != [batch_size, 1, tgt_len, src_len] {
                panic!(
                    "attn_mask should be of shape {:?}, but is {:?}",
                    [batch_size, 1, tgt_len, src_len],
                    attn_mask.dims()
                );
            }
            let masked_attn_weights = attn_weights
                .reshape([batch_size, self.num_heads, tgt_len, src_len])
                .add(attn_mask);
            attn_weights =
                masked_attn_weights.reshape([batch_size * self.num_heads, tgt_len, src_len]);
            // {
            //     println!("attn_weights: {:?}", attn_weights.shape());
            //     let out = attn_weights.to_string();
            //     let out = out.split(":").collect::<Vec<&str>>()[1].trim().to_string();
            //     let out = out.split(",\n  shape").collect::<Vec<&str>>()[0]
            //         .trim()
            //         .to_string();
            //     println!("{}", out);
            // }
        }

        attn_weights = softmax(attn_weights, 2);
        attn_weights = attn_weights.matmul(value_states);

        if attn_weights.dims() != [batch_size * self.num_heads, tgt_len, self.head_dim] {
            panic!(
                "attn_output should be of shape {:?}, but is {:?}",
                [batch_size * self.num_heads, tgt_len, self.head_dim],
                attn_weights.dims()
            );
        }

        let attn_output = attn_weights
            .reshape([batch_size, self.num_heads, tgt_len, self.head_dim])
            .swap_dims(1, 2)
            .reshape([batch_size, tgt_len, self.embed_dim]);

        self.out_proj.forward(attn_output)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2FeedForward<B: Backend> {
    pub intermediate_dropout: Dropout,
    pub intermediate_dense: Linear<B>,
    pub intermediate_act_fn: Gelu,
    pub output_dense: Linear<B>,
    pub output_dropout: Dropout,
}

impl<B: Backend> Wav2Vec2FeedForward<B> {
    pub fn init(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        activation_dropout: f64,
        hidden_dropout: f64,
        device: &Device<B>,
    ) -> Self {
        let intermediate_dropout = DropoutConfig::new(activation_dropout).init();
        let intermediate_dense = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(bias)
            .init(device);
        let intermediate_act_fn = Gelu::new();
        let output_dense = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(bias)
            .init(device);
        let output_dropout = DropoutConfig::new(hidden_dropout).init();
        Self {
            intermediate_dropout,
            intermediate_dense,
            intermediate_act_fn,
            output_dense,
            output_dropout,
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.intermediate_dense.forward(hidden_states);
        let hidden_states = self.intermediate_act_fn.forward(hidden_states);
        let hidden_states = self.intermediate_dropout.forward(hidden_states);

        let hidden_states = self.output_dense.forward(hidden_states);
        self.output_dropout.forward(hidden_states)
    }
}
