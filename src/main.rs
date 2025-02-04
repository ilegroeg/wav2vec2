use std::time::Instant;

use burn::backend::LibTorch;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use wav2vec2::components::config::Wav2Vec2Config;
use wav2vec2::components::ctc::Wav2Vec2ForCTC;
use wav2vec2::components::processor::Padding;
use wav2vec2::components::processor::PaddingSide;
use wav2vec2::components::processor::PaddingStrategy;
use wav2vec2::components::processor::RawAudio;
use wav2vec2::components::processor::SampleRateHz;
use wav2vec2::components::processor::Wav2Vec2FeatureExtractor;
use wav2vec2::components::processor::Wav2Vec2Tokenizer;
use wav2vec2::components::util::float_tensor_to_string_pretty;
use wav2vec2::components::util::load_model;
use wav2vec2::components::util::load_weights_from_torch;
use wav2vec2::components::util::write_to_file;

const _CONFIG_PATH: &str = "./src/configs/wav2vec2-base-wonders-phonemes.json";

fn _benchmark_model<B: Backend>(n_runs: usize, device: &Device<B>) {
    let config = Wav2Vec2Config::from_json_file(_CONFIG_PATH).unwrap();
    println!("config: {:#?}", config);
    let model = Wav2Vec2ForCTC::init(&config, device);
    let dims: Vec<usize> = vec![1, 44749]; // [batch, length]
    let input =
        Tensor::<B, 2>::random(dims, burn::tensor::Distribution::Uniform(-1.0, 1.0), device);
    // Warmup runs
    println!("Warming up...");
    for i in 0..5 {
        let input = input.clone();
        let _out = model.forward(input, None);
        println!("Warmup run {}/5", i + 1);
    }

    // Benchmark runs
    println!("\nStarting benchmark...");
    let mut total_duration = std::time::Duration::new(0, 0);
    let mut durations = Vec::new();
    for i in 0..n_runs {
        let input = input.clone();

        let start = Instant::now();
        let _out = model.forward(input, None);
        let duration = start.elapsed();

        total_duration += duration;
        durations.push(duration);
        println!("Run {}/{}: {:?}", i + 1, n_runs, duration);
    }

    let avg_duration = total_duration.as_secs_f64() / n_runs as f64;
    let std_dev = durations
        .iter()
        .map(|d| (d.as_secs_f64() - avg_duration).powi(2))
        .sum::<f64>()
        / n_runs as f64;
    let std_dev = std_dev.sqrt();
    println!(
        "durations: {:?}",
        durations
            .iter()
            .map(|d| d.as_secs_f64())
            .collect::<Vec<f64>>()
    );
    println!("\nResults:");
    println!("Average forward pass time: {:.4?} seconds", avg_duration);
    println!("Std dev of forward pass time: {:.4?} seconds", std_dev);
}

fn _process_audio<B: Backend>(
    device: &Device<B>,
) -> (Tensor<B, 2>, Option<Tensor<B, 2, burn::tensor::Int>>) {
    let file_content =
        std::fs::read_to_string("./src/recordings/long_stream_poll_sample.json").unwrap();
    let ragged_array: Vec<Vec<u8>> = serde_json::from_str(&file_content).unwrap();
    let mut audios = Vec::<RawAudio>::with_capacity(3);
    for i in 0..ragged_array.len() {
        audios.push(RawAudio {
            data: ragged_array[i].clone(),
            sample_rate: SampleRateHz::from_hz(8000),
        });
        let shape = [audios[i].data.len()];
        let audio_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(audios[i].data.clone(), shape), device);
        println!(
            "Audio {i}:\n{}",
            float_tensor_to_string_pretty(&audio_tensor)
        );
        println!("{:?}\n", audio_tensor.shape());
    }

    let feature_extractor = Wav2Vec2FeatureExtractor::new(
        SampleRateHz::from_hz(8000),
        Padding::new(0.0, PaddingStrategy::Longest, false, PaddingSide::Right),
    );

    let (processed_audios, attention_masks) =
        feature_extractor.prepare_audio::<B>(&audios, true, device);
    println!(
        "batched audios:\n {}",
        float_tensor_to_string_pretty(&processed_audios)
    );
    println!("{:?}\n", processed_audios.shape());
    if let Some(attention_mask) = attention_masks.clone() {
        let attention_mask = attention_mask.to_data();
        let attention_mask = Tensor::<B, 2, burn::tensor::Float>::from_data(attention_mask, device);
        println!(
            "atn mask:\n{}",
            float_tensor_to_string_pretty(&attention_mask)
        );
        println!(
            "check lens:\n{}",
            float_tensor_to_string_pretty(&attention_mask.clone().sum_dim(1))
        );
        println!("{:?}\n", attention_mask.shape());
    }

    (processed_audios, attention_masks)
}

fn _test_individual_components<B: Backend>(device: &Device<B>) {
    let (processed_audios, _attn_mask) = _process_audio::<B>(device);

    let model = load_model::<B>(None, device);

    // let feature_extractor = model.wav2vec2.feature_extractor.clone();
    // let feature_projection = model.wav2vec2.feature_projection.clone();
    // let pos_embed = model.wav2vec2.encoder.pos_conv_embed.clone();
    // let layer_norm = model.wav2vec2.encoder.layer_norm;
    // let dropout = model.wav2vec2.encoder.dropout.clone();
    // let pos_embed_conv = pos_embed.conv;

    // let layer1 = model.wav2vec2.encoder.layers[0].clone();
    // let encoder = model.wav2vec2.encoder.clone();

    // let out = feature_extractor.forward(processed_audios);
    // println!("out shape: {:?}", out.shape());
    // let out = out.swap_dims(1, 2);
    // println!("out shape: {:?}", out.shape());
    // let attn_mask = attn_mask.unwrap();

    // let out_lens = model.wav2vec2.get_feature_extract_output_lengths(attn_mask.clone().sum_dim(1).squeeze(1));

    // let out_mask = model.wav2vec2.reduce_attn_mask(out.dims()[1], attn_mask.clone(), device);
    // let out_mask_dims = out_mask.shape();
    // println!("{:?}", out_mask);

    // let out_mask_sum = out_mask.clone().sum_dim(1);

    // let attn_mask = {
    //     // (batch_size, 1, 1, seqlen)
    //     let reshaped_mask = out_mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
    //     let negated_mask = Tensor::ones_like(&reshaped_mask).sub(reshaped_mask);
    //     // torch uses f32 as the standard float type for tensors
    //     let lnn_mask = negated_mask.mul_scalar(f32::MIN);
    //     lnn_mask.expand([out_mask_dims.dims[0], 1, out_mask_dims.dims[1], out_mask_dims.dims[1]])
    // };
    // println!("mask shape: {:?}", attn_mask.shape());

    // let (out, _) = feature_projection.forward(out);
    // println!("{:?}", out.shape());

    // let v = pos_embed.weight_v.val();
    // println!("v shape: {:?}", v.shape());
    // let g = pos_embed.weight_g.val();
    // println!("g shape: {:?}", g.shape());

    // let norm_v = norm(v, NormType::L2, vec![0, 1]);
    // println!("norm_v shape: {:?}", norm_v.shape());
    // println!("{}", float_tensor_to_string_pretty(&norm_v));

    // let out = weight_norm.forward(v, g);
    // println!("out shape: {:?}", out.shape());
    // println!("{}", float_tensor_to_string_pretty(&out));

    // let pos_embed_out = pos_embed.forward(out.clone());
    // println!("{:?}", pos_embed_out.shape());
    // let out = out.add(pos_embed_out);
    // println!("{:?}", out.shape());
    // let out = layer_norm.forward(out);
    // println!("{:?}", out.shape());
    // let out = dropout.forward(out);
    // println!("{:?}", out.shape());
    // println!("{}", float_tensor_to_string_pretty(&out));

    // let q = layer1.attention.q_proj.weight;
    // let k = layer1.attention.k_proj.weight;
    // let v = layer1.attention.v_proj.weight;
    // println!("{:?}", q.shape());
    // println!("q: {}", float_tensor_to_string_pretty(&q));
    // println!("{:?}", k.shape());
    // println!("k: {}", float_tensor_to_string_pretty(&k));
    // println!("{:?}", v.shape());
    // println!("v: {}", float_tensor_to_string_pretty(&v));

    // let out = layer1.forward(out, Some(attn_mask.clone()));
    // println!("{:?}", out.shape());
    // println!("{}", float_tensor_to_string_pretty(&out));

    // let out = encoder.forward(out, Some(out_mask.clone()));
    // println!("{:?}", out.shape());
    // println!("{}", float_tensor_to_string_pretty(&out));

    // attention mask does not work with Candle (doesn't allow for tensors to be divded by scalars)
    // works with LibTorch
    let out = model.forward(processed_audios, _attn_mask);
    println!("logits:\n{:?}", out.shape());
    println!("{}\n", float_tensor_to_string_pretty(&out));

    let batched_token_ids = out.argmax(2);

    let tokenizer = Wav2Vec2Tokenizer::new();
    let batched_phonemes = tokenizer.decode(&batched_token_ids);
    for (i, batch) in batched_phonemes.iter().enumerate() {
        println!("phoneme {i}:\n{}", batch.0);
    }

    // save to file
    write_to_file(
        "./src/recordings/phonemes_w_attn_mask.txt",
        &batched_phonemes,
    );
}

type BE = LibTorch;

fn main() {
    let device = Default::default();
    // _computation::<BE>(&device);
    // _benchmark_model::<BE>(10, &device);
    load_weights_from_torch::<BE>(false, &device);
    // let _ = load_model::<BE>(None, &device);
    // _process_audio::<BE>(&device);
    // _test_individual_components::<BE>(&device);
}
