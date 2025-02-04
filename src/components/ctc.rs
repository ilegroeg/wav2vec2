// Wav2Vec2 Model with a language modeling head on top
// for Connectionist Temporal Classification (CTC).
// Read more about CTC at: https://distill.pub/2017/ctc/
// Read more about Wav2Vec2 at: https://arxiv.org/abs/2006.11477

use burn::module::Module;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::prelude::Backend;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::TensorData;

use crate::components::config::Wav2Vec2Config;
use crate::components::encoder::Wav2Vec2EncoderStableLayerNorm;
use crate::components::feature_extractor::Wav2Vec2FeatureEncoder;
use crate::components::feature_extractor::Wav2Vec2FeatureProjection;
#[derive(Module, Debug)]
pub struct Wav2Vec2ForCTC<B: Backend> {
    pub wav2vec2: Wav2Vec2Model<B>,
    pub dropout: Dropout,
    pub lm_head: Linear<B>,
}

impl<B: Backend> Wav2Vec2ForCTC<B> {
    pub fn init(config: &Wav2Vec2Config, device: &Device<B>) -> Self {
        let wav2vec2 = Wav2Vec2Model::init(config, device);
        let dropout = DropoutConfig::new(config.final_dropout).init();
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size).init(device);
        Self {
            wav2vec2,
            dropout,
            lm_head,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        attn_mask: Option<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 3> {
        let outputs = self.wav2vec2.forward(input, attn_mask);
        let outputs = self.dropout.forward(outputs);
        self.lm_head.forward(outputs)
    }
}

#[derive(Module, Debug)]
pub struct Wav2Vec2Model<B: Backend> {
    pub feature_extractor: Wav2Vec2FeatureEncoder<B>,
    pub feature_projection: Wav2Vec2FeatureProjection<B>,
    pub encoder: Wav2Vec2EncoderStableLayerNorm<B>,
}

impl<B: Backend> Wav2Vec2Model<B> {
    pub fn init(config: &Wav2Vec2Config, device: &Device<B>) -> Self {
        let feature_extractor = Wav2Vec2FeatureEncoder::init(config, device);
        let feature_projection = Wav2Vec2FeatureProjection::init(
            config.conv_dim[config.conv_dim.len() - 1],
            config.hidden_size,
            config.feat_proj_dropout,
            device,
        );
        let encoder = Wav2Vec2EncoderStableLayerNorm::init(config, device);
        Self {
            feature_extractor,
            feature_projection,
            encoder,
        }
    }

    pub fn get_feature_extract_output_lengths(
        &self,
        input_lengths: Tensor<B, 1, burn::tensor::Int>,
    ) -> Tensor<B, 1, burn::tensor::Int> {
        let mut lengths = input_lengths;
        for i in 0..self.feature_extractor.conv_layers.len() {
            // conv out len calcs
            // https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            lengths = lengths
                .sub_scalar(self.feature_extractor.conv_layers[i].conv().kernel_size as u8)
                .div_scalar(self.feature_extractor.conv_layers[i].conv().stride as u8)
                .add_scalar(1);
        }
        lengths
    }

    pub fn reduce_attn_mask(
        &self,
        seq_len: usize,
        attn_mask: Tensor<B, 2, burn::tensor::Int>,
        device: &Device<B>,
    ) -> Tensor<B, 2> {
        let batch_size = attn_mask.dims()[0];
        let non_padded_lengths = attn_mask.sum_dim(1).squeeze(1);
        let final_lengths = self.get_feature_extract_output_lengths(non_padded_lengths);
        let final_lengths_data = final_lengths.to_data().to_vec::<i64>().unwrap();

        let mut final_ones: Vec<f32> = Vec::with_capacity(batch_size * seq_len);
        for &length in final_lengths_data.iter() {
            let mut row = vec![0.0; seq_len];
            row[..length as usize].fill(1.0);
            final_ones.extend(row);
        }
        let final_ones_data = TensorData::new(final_ones, [batch_size, seq_len]);
        Tensor::from_data(final_ones_data, device)
    }

    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        attn_mask: Option<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 3> {
        let device = &input.device();
        let extracted_features = self.feature_extractor.forward(input);
        let extracted_features = extracted_features.swap_dims(1, 2);

        let attn_mask =
            attn_mask.map(|mask| self.reduce_attn_mask(extracted_features.dims()[1], mask, device));

        let (hidden_states, _extracted_features) =
            self.feature_projection.forward(extracted_features);
        self.encoder.forward(hidden_states, attn_mask)
    }
}
