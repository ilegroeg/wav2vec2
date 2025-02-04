use std::collections::HashMap;
use std::fmt::Display;

use burn::prelude::Backend;
use burn::tensor::Device;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
#[derive(Default)]
pub struct Wav2Vec2Tokenizer {
    pub padding_token: String,
    pub _word_delimiter_token: String,
    pub replace_word_delimiter_char: String,
    pub encoder: HashMap<String, i64>,
    pub decoder: HashMap<i64, String>,
}

impl Wav2Vec2Tokenizer {
    pub fn new() -> Self {
        let vocab = include_str!("../configs/vocab.json");
        let vocab: HashMap<String, usize> = serde_json::from_str(vocab)
            .map_err(|e| format!("Failed to parse vocab.json: {}", e))
            .unwrap();

        let padding_token = "[PAD]".to_string();
        let _word_delimiter_token = "|".to_string();
        let replace_word_delimiter_char = " ".to_string();

        // Create encoder/decoder with proper UTF-8 handling
        let encoder = vocab
            .iter()
            .map(|(k, v)| (k.chars().collect::<String>(), *v as i64))
            .collect();
        let decoder = vocab
            .iter()
            .map(|(k, v)| (*v as i64, k.chars().collect::<String>()))
            .collect();

        Self {
            padding_token,
            _word_delimiter_token,
            replace_word_delimiter_char,
            encoder,
            decoder,
        }
    }

    pub fn decode<B: Backend>(
        &self,
        batched_token_ids: &Tensor<B, 3, burn::tensor::Int>,
    ) -> Vec<Phonemes> {
        // Tensor to vec
        let batch_size = batched_token_ids.dims()[0];
        let flat_token_ids = batched_token_ids.to_data().to_vec::<i64>().unwrap();
        let single_input_size = flat_token_ids.len() / batch_size;
        let batched_token_ids = flat_token_ids
            .chunks(single_input_size)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<i64>>>();

        // ids to tokens
        let tokens = batched_token_ids
            .iter()
            .map(|batch| self.ids_to_tokens(batch))
            .collect::<Vec<Vec<String>>>();

        // collapse with ctc
        tokens
            .into_iter()
            .map(|batch| self.tokens_to_string(batch))
            .collect::<Vec<Phonemes>>()
    }

    pub fn ids_to_tokens(&self, ids: &[i64]) -> Vec<String> {
        let mut tokens = Vec::new();
        for id in ids {
            if *id >= self.decoder.len() as i64 {
                println!("id: {}, not in vocab check your model's final layer", id);
                continue;
            }
            tokens.push(self.decoder.get(id).unwrap().clone());
        }
        tokens
    }

    pub fn tokens_to_string(&self, tokens: Vec<String>) -> Phonemes {
        let mut grouped_tokens = Vec::new();
        let mut counts = Vec::new();
        let mut current = tokens.first().cloned();
        let mut count = 0;

        for token in tokens {
            if Some(&token) == current.as_ref() {
                count += 1;
            } else {
                if let Some(t) = current {
                    grouped_tokens.push(t);
                    counts.push(count);
                }
                current = Some(token);
                count = 1;
            }
        }

        if let Some(t) = current {
            grouped_tokens.push(t);
            counts.push(count);
        }

        //remove padding token
        let grouped_tokens_no_padding = grouped_tokens
            .into_iter()
            .filter(|token| *token != self.padding_token)
            .collect::<Vec<String>>();
        Phonemes::from_string(grouped_tokens_no_padding.join(""))
    }
}

pub struct Wav2Vec2FeatureExtractor {
    sampling_rate: SampleRateHz,
    padding: Padding,
}

impl Wav2Vec2FeatureExtractor {
    pub fn new(sampling_rate: SampleRateHz, padding: Padding) -> Self {
        Self {
            sampling_rate,
            padding,
        }
    }

    pub fn prepare_audio<B: Backend>(
        &self,
        audios: &[RawAudio],
        // no attention mask works better for group-norm model arch
        return_attention_mask: bool,
        device: &Device<B>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2, Int>>) {
        // check if all audios have the same sampling rate as the processor
        for audio in audios {
            if audio.sample_rate != self.sampling_rate {
                panic!("All audios must have the same sampling rate as the processor");
            }
        }

        let max_audio_length = audios.iter().map(|x| x.data.len()).max().unwrap();
        let max_length = match self.padding.padding_strategy {
            PaddingStrategy::Longest => max_audio_length,
            PaddingStrategy::MaxLength(length) => length,
            PaddingStrategy::PadToMultiple(multiple) => {
                (max_audio_length / multiple + 1) * multiple
            }
        };

        if max_length < max_audio_length && !self.padding.truncate {
            panic!("Max length is less than the max audio length and truncation is not allowed");
        }

        let float_audios: Vec<Vec<f32>> = audios
            .iter()
            .map(|raw_audio| {
                raw_audio
                    .data
                    .iter()
                    // Each &u8 is mapped to a float. Adjust however you want:
                    .map(|&byte| byte as f32)
                    .collect::<Vec<f32>>()
            })
            .collect();

        let (prepared_audio, attention_mask) = if return_attention_mask {
            let shape = [float_audios.len(), max_length];
            let mut data = Vec::with_capacity(shape.iter().product());
            for audio in &float_audios {
                let rest = max_length - audio.len();
                // add 1 for the audio data and 0 for the padding
                data.extend(std::iter::repeat(1).take(audio.len()));
                data.extend(std::iter::repeat(0).take(rest));
            }

            let data = TensorData::new(data, shape);
            let attention_mask = Tensor::<B, 2, Int>::from_data(data, device);

            let normalized_audio = float_audios
                .iter()
                .map(|audio| self.normalize_audio(audio))
                .collect::<Vec<Vec<f32>>>();

            let padded_audio = normalized_audio
                .iter()
                .map(|audio| self.pad_audio(audio, max_length))
                .collect::<Vec<Vec<f32>>>();

            (padded_audio, Some(attention_mask))
        } else {
            let attention_mask = None;

            // fit the audios in a matrix (requires unit length)
            let padded_audio = float_audios
                .iter()
                .map(|audio| self.pad_audio(audio, max_length))
                .collect::<Vec<Vec<f32>>>();

            // for better performance
            let normalized_audio = padded_audio
                .iter()
                .map(|audio| self.normalize_audio(audio))
                .collect::<Vec<Vec<f32>>>();

            (normalized_audio, attention_mask)
        };

        // burn is requires data (to be passed to tensor) to be next to each other in memory
        let shape = [prepared_audio.len(), prepared_audio[0].len()];
        let flattened = prepared_audio.into_iter().flatten().collect();
        let data = TensorData::new(flattened, shape);

        let prepared_audio_tensor = Tensor::<B, 2>::from_data(data, device);

        (prepared_audio_tensor, attention_mask)
    }

    // zero mean unit variance
    fn normalize_audio(&self, data: &[f32]) -> Vec<f32> {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        data.iter()
            // 1e-7 is added to match the pytorch implementation
            .map(|x| (x - mean) / (variance + 1e-7).sqrt())
            .collect()
    }

    fn pad_audio(&self, data: &[f32], max_length: usize) -> Vec<f32> {
        let padding_needed = max_length - data.len();
        let mut padded: Vec<f32> = Vec::with_capacity(max_length);
        match self.padding.side {
            PaddingSide::Left => {
                padded.extend(std::iter::repeat(self.padding.padding_value).take(padding_needed));
                padded.extend_from_slice(data);
            }
            PaddingSide::Right => {
                padded.extend_from_slice(data);
                padded.extend(std::iter::repeat(self.padding.padding_value).take(padding_needed));
            }
        }
        padded
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleRateHz(u32);

impl SampleRateHz {
    pub fn from_hz(hz: u32) -> Self {
        Self(hz)
    }
}

#[derive(Clone)]
pub struct RawAudio {
    pub data: Vec<u8>,
    pub sample_rate: SampleRateHz,
}

pub enum PaddingStrategy {
    // if audio length can fit in memory
    Longest,
    MaxLength(usize),
    // For more efficient tensor core utilization (need to look more into this)
    PadToMultiple(usize),
}

pub enum PaddingSide {
    Left,
    Right,
}

pub struct Padding {
    pub padding_value: f32,
    pub padding_strategy: PaddingStrategy,
    pub truncate: bool,
    pub side: PaddingSide,
}

impl Padding {
    pub fn new(
        padding_value: f32,
        padding_strategy: PaddingStrategy,
        truncate: bool,
        side: PaddingSide,
    ) -> Self {
        Self {
            padding_value,
            padding_strategy,
            truncate,
            side,
        }
    }
}

pub struct Phonemes(pub String);

impl Display for Phonemes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Phonemes {
    pub fn from_string(phonemes: String) -> Self {
        Self(phonemes)
    }
}
