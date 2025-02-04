use serde::Deserialize;
use serde::Serialize;

use crate::components::processor::SampleRateHz;

#[derive(Serialize, Deserialize, Debug)]
pub struct Wav2Vec2Config {
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub conv_bias: bool,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub final_dropout: f64,
    pub feat_extract_norm: String,
    pub feat_proj_dropout: f64,
    pub hidden_dropout: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub num_conv_pos_embeddings: usize,
    pub num_feat_extract_layers: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
}

impl Wav2Vec2Config {
    pub fn from_json_file(path: &str) -> Result<Self, std::io::Error> {
        let json_str = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json_str)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeModelConfig {
    /// Number of indenpendent models to run in parallel
    pub n_models: usize,
    /// For each model instance, how many threads do we use
    pub per_model_parallelism: usize,
    /// How many to dequeue at a time for each model
    pub per_model_max_batch_size: usize,
    /// sample rate of the audio
    pub sample_rate: SampleRateHz,
}

impl Default for PhonemeModelConfig {
    fn default() -> Self {
        Self {
            n_models: 1,
            per_model_parallelism: 4,
            per_model_max_batch_size: 32,
            sample_rate: SampleRateHz::from_hz(16000),
        }
    }
}

impl PhonemeModelConfig {
    pub fn new() -> Self {
        Self::default()
    }
}
