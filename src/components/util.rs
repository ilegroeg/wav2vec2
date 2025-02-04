use burn::config::Config;
use burn::module::Ignored;
use burn::module::Module;
use burn::module::Param;
use burn::nn::Initializer;
use burn::nn::PaddingConfig1d;
use burn::prelude::Backend;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::ops::conv::calculate_conv_padding;

#[derive(Config, Debug)]
pub struct ParameterizedConv1dConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    pub kernel_size: usize,
    // which dim to keep)
    #[config(default = "None")]
    pub weight_norm_dim: Option<usize>,
    #[config(default = "1")]
    pub stride: usize,
    // Spacing between kernel elements.
    #[config(default = "1")]
    pub dilation: usize,
    // Controls the connections between input and output channels.
    #[config(default = "1")]
    pub groups: usize,
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    #[config(default = true)]
    pub bias: bool,
    // The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl ParameterizedConv1dConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ParameterizedConv1d<B> {
        let shape_g = [1, 1, self.kernel_size];
        let shape_v = [
            self.channels_out,
            self.channels_in / self.groups,
            self.kernel_size,
        ];

        let fan_in: usize = self.channels_in / self.groups * self.kernel_size;
        let weight_g = self
            .initializer
            .init_with(shape_g, Some(fan_in), None, device);
        let weight_v = self
            .initializer
            .init_with(shape_v, Some(fan_in), None, device);

        let mut bias = None;

        if self.bias {
            bias =
                Some(
                    self.initializer
                        .init_with([self.channels_out], Some(fan_in), None, device),
                );
        }

        let weight_norm_dim = if let Some(dim) = self.weight_norm_dim {
            dim as i32
        } else {
            -1
        };

        ParameterizedConv1d {
            weight_g,
            weight_v,
            weight_norm_dim,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

#[derive(Module, Debug)]
pub struct ParameterizedConv1d<B: Backend> {
    // Tensor of shape `[1, 1, kernel_size]` (original0)
    pub weight_g: Param<Tensor<B, 3>>,
    // Tensor of shape `[channels_out, channels_in / groups, kernel_size]` (original1)
    pub weight_v: Param<Tensor<B, 3>>,
    pub weight_norm_dim: i32,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub stride: usize,
    pub kernel_size: usize,
    // Spacing between kernel elements.
    pub dilation: usize,
    // Controls the connections between input and output channels.
    pub groups: usize,
    pub padding: Ignored<PaddingConfig1d>,
}

impl<B: Backend> ParameterizedConv1d<B> {
    // implemented from pytorch's weight norm https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html
    // opted out of using struct for weight_norm since it caused loag_args to fail
    pub fn _weight_norm(&self, weight_v: Tensor<B, 3>, weight_g: Tensor<B, 3>) -> Tensor<B, 3> {
        let rank = weight_v.dims().len();
        let mut axes = (0..rank).collect::<Vec<_>>();
        if self.weight_norm_dim != -1 {
            axes.remove(self.weight_norm_dim as usize);
        }
        let norm_v = norm(weight_v.clone(), NormType::L2, axes);
        let weight_v = weight_v.div(norm_v);
        weight_v.mul(weight_g)
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch_size, _channels, length] = input.dims();
        let padding = match *self.padding {
            PaddingConfig1d::Valid => 0,
            PaddingConfig1d::Same => {
                calculate_conv_padding(self.kernel_size, self.stride, length, length)
            }
            PaddingConfig1d::Explicit(value) => value,
        };

        conv1d(
            input,
            self._weight_norm(self.weight_v.val(), self.weight_g.val()),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new([self.stride], [padding], [self.dilation], self.groups),
        )
    }
}

pub enum NormType {
    L1,
    L2,
}

pub fn norm<B: Backend>(mut weight_v: Tensor<B, 3>, p: NormType, axes: Vec<usize>) -> Tensor<B, 3> {
    match p {
        NormType::L1 => {
            weight_v = weight_v.abs();
            for axis in axes {
                weight_v = weight_v.sum_dim(axis);
            }
            weight_v
        }
        NormType::L2 => {
            weight_v = weight_v.powf_scalar(2);
            for axis in axes {
                weight_v = weight_v.sum_dim(axis);
            }
            weight_v.sqrt()
        }
    }
}

// use burn::backend::LibTorch;
// use burn::module::Module;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;
use burn::record::Recorder;
// use burn::tensor::Device;
// use burn::tensor::Tensor;
// use burn::tensor::TensorData;
// use burn::tensor::backend::Backend;
use burn_import::pytorch::LoadArgs;
use burn_import::pytorch::PyTorchFileRecorder;

use crate::components::config::Wav2Vec2Config;
use crate::components::ctc::Wav2Vec2ForCTC;

const _TORCH_WEIGHTS: &str = "src/external_weights/model-02042025.pth";
const MODEL_PATH: &str = "src/model_binaries/model-02042025.mpk";
const CONFIG_PATH: &str = "src/configs/wav2vec2-base-group.json";

pub fn load_weights_from_torch<B: Backend>(save_to_binary: bool, device: &Device<B>) {
    let config = Wav2Vec2Config::from_json_file(CONFIG_PATH).unwrap();
    let model: Wav2Vec2ForCTC<B> = Wav2Vec2ForCTC::init(&config, device);

    // Load weights from torch state_dict
    let load_args = LoadArgs::new(_TORCH_WEIGHTS.into())
        .with_debug_print()
        .with_key_remap(
            "(.+)\\.parametrizations\\.weight\\.original0",
            "$1.weight_g",
        )
        .with_key_remap(
            "(.+)\\.parametrizations\\.weight\\.original1",
            "$1.weight_v",
        );

    println!("\n=== Attempting to load weights ===");
    match PyTorchFileRecorder::<FullPrecisionSettings>::default().load(load_args, device) {
        Ok(record) => {
            let _model = model.load_record(record);
            println!("Successfully loaded weights!");
            if save_to_binary {
                // Save the model to a supported format and load it back
                let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
                _model
                    .clone() // `save_file` takes ownership but we want to load the file after
                    .save_file(MODEL_PATH, &recorder)
                    .expect("Should be able to save weights to file");
                println!("Successfully saved weights to binary file");

                load_model::<B>(None, device);
            }
        }
        Err(e) => {
            println!("Failed to load weights with error: {:?}", e);
            // Print the first few characters of the error message to see exactly where it's failing
            if let Some(err_str) = e.to_string().get(..200) {
                println!("Error preview: {}", err_str);
            }
        }
    }
}

/// Load the model from the file in your source code (not in build.rs or script).
pub fn load_model<B: Backend>(
    path_to_model_binary: Option<&str>,
    device: &Device<B>,
) -> Wav2Vec2ForCTC<B> {
    println!("Loading model from binary");
    let record = match path_to_model_binary {
        Some(path) => NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(path.into(), device)
            .expect("Should decode state successfully"),
        None => NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(MODEL_PATH.into(), device)
            .expect("Should decode state successfully"),
    };

    let config = Wav2Vec2Config::from_json_file(CONFIG_PATH).unwrap();
    let model = Wav2Vec2ForCTC::<B>::init(&config, device).load_record(record);
    println!("Successfully loaded weights from binary file");
    model
}

pub fn float_tensor_to_string_pretty<B: Backend, const D: usize>(t: &Tensor<B, D>) -> String {
    let out = t.to_string();
    let out = out.split(":").collect::<Vec<&str>>()[1].trim().to_string();
    let out = out.split(",\n  shape").collect::<Vec<&str>>()[0]
        .trim()
        .to_string();
    out
}

use std::io::Write;
pub fn write_to_file<T: std::fmt::Display>(file_path: &str, data: &[T]) {
    // save to file as list
    let mut file = std::fs::File::create(file_path).unwrap();
    for batch in data.iter() {
        writeln!(file, "{}", batch).unwrap();
    }
}
