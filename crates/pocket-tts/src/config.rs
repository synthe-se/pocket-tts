//! Configuration types for pocket-tts, matching Python's utils/config.py

use serde::Deserialize;
use std::path::Path;

/// Flow network configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    /// Sampler head: "lsd" (two time conditions, 1-step decode; every
    /// released model), "flow_matching" (one time condition, Euler
    /// integration, wants >= 16 decode steps) or "drifting" (no time
    /// condition, the head maps noise to a sample in one step).
    #[serde(default, rename = "type")]
    pub flow_type: FlowType,
}

/// The flow head's sampling objective (upstream `FlowConfig.type`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlowType {
    #[default]
    Lsd,
    FlowMatching,
    Drifting,
}

impl FlowType {
    /// Number of time conditions the head is trained with.
    pub fn num_time_conds(self) -> usize {
        match self {
            FlowType::Lsd => 2,
            FlowType::FlowMatching => 1,
            FlowType::Drifting => 0,
        }
    }
}

/// Transformer configuration for FlowLM
#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

/// Lookup table (text conditioner) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

/// FlowLM model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMConfig {
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    /// When true, a learnt BOS embedding (`bos_before_voice`) is prepended to
    /// the voice conditioning before it is fed to the FlowLM transformer.
    /// Required by the multilingual checkpoints (e.g. `french_24l`).
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

/// SEANet encoder/decoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SEANetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

/// Transformer configuration for Mimi
#[derive(Debug, Clone, Deserialize)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: usize,
}

fn default_max_period() -> f64 {
    10000.0
}

/// Quantizer configuration
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

/// Mimi model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct MimiConfig {
    pub dtype: String,
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f64,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    /// "v2 of models": output dimension of the encoder-side `ConvDownsample1d`
    /// (the unquantized latent dimension). `None` falls back to the SEANet
    /// dimension, matching the pre-multilingual checkpoints.
    #[serde(default)]
    pub inner_dim: Option<usize>,
    /// "v2 of models": input dimension of the decoder-side `ConvTrUpsample1d`.
    /// `None` falls back to the SEANet dimension.
    #[serde(default)]
    pub outer_dim: Option<usize>,
}

/// Root configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
    /// Prepend 8 spaces to very short inputs (< 5 words). The English
    /// checkpoints rely on this; the multilingual ones do not.
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    /// Replace `;` with `,` before tokenisation (multilingual checkpoints).
    #[serde(default)]
    pub remove_semicolons: bool,
    /// Make sure the prompt ends with sentence-final punctuation (#296).
    #[serde(default = "default_true")]
    pub append_terminal_punctuation: bool,
    /// Upper-case the first letter of the prompt. Models whose text is
    /// romanised phonemes switch it off: the capital is not in their
    /// inventory (#c6f0aac).
    #[serde(default = "default_true")]
    pub capitalize_first_letter: bool,
    /// Per-character rewrites applied before tokenization ("" deletes), for
    /// characters the model's training text never contained (#497f399).
    #[serde(default)]
    pub replace_characters: std::collections::HashMap<String, String>,
    /// Model-recommended number of frames to keep generating after EOS.
    /// Overrides the heuristic when set.
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<usize>,
    /// Model-recommended sampling temperature, used when the caller does not
    /// pass one. 0.3 beats 0.7 on WER and UTMOS for every shipped model
    /// (#223, #324); a config sets its own value only if tuned elsewhere.
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
}

fn default_temperature() -> f32 {
    defaults::TEMPERATURE
}

fn default_true() -> bool {
    true
}

/// Load configuration from a YAML file
pub fn load_config<P: AsRef<Path>>(path: P) -> anyhow::Result<Config> {
    let contents = std::fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&contents)?;
    Ok(config)
}

/// Default generation parameters (matching Python's default_parameters.py)
pub mod defaults {
    pub const TEMPERATURE: f32 = 0.3;
    /// Upstream DEFAULT_SAMPLER_DECODE_STEPS (was DEFAULT_LSD_DECODE_STEPS).
    pub const SAMPLER_DECODE_STEPS: usize = 1;
    /// Former name of [`SAMPLER_DECODE_STEPS`], kept for callers.
    pub const LSD_DECODE_STEPS: usize = SAMPLER_DECODE_STEPS;
    pub const NOISE_CLAMP: Option<f32> = None;
    pub const EOS_THRESHOLD: f32 = -4.0;
    /// Upstream DEFAULT_LANGUAGE. The pre-language checkpoint "b6369a24"
    /// (= english_2026-01) stays available as a variant.
    pub const DEFAULT_VARIANT: &str = "english";
    // TODO(upstream): make this dynamic since english_2026-04 supports bigger chunks
    pub const MAX_TOKEN_PER_CHUNK: usize = 50;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("pocket_tts")
            .join("config")
            .join("b6369a24.yaml")
    }

    fn shipped_configs() -> Vec<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config");
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn test_shipped_configs_sample_at_the_tuned_temperature() {
        // Upstream #322/#324: every shipped model samples at 0.3.
        let paths = shipped_configs();
        assert!(paths.len() >= 20);
        for path in paths {
            let config = load_config(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            assert_eq!(config.default_temperature, 0.3, "{path:?}");
            crate::text_chunking::TextRules::from_config(&config)
                .unwrap_or_else(|e| panic!("{path:?}: {e}"));
        }
    }

    #[test]
    fn test_flow_types() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config");
        let drifting = load_config(dir.join("english_drifting_26-09.yaml")).unwrap();
        assert_eq!(drifting.flow_lm.flow.flow_type, FlowType::Drifting);
        assert_eq!(FlowType::Drifting.num_time_conds(), 0);
        let lsd = load_config(dir.join("english.yaml")).unwrap();
        assert_eq!(lsd.flow_lm.flow.flow_type, FlowType::Lsd);
        let french = load_config(dir.join("french.yaml")).unwrap();
        assert_eq!(
            french.replace_characters.get("\u{ab}").map(String::as_str),
            Some("")
        );
        assert!(french.capitalize_first_letter && french.append_terminal_punctuation);
    }

    #[test]
    fn test_load_config() {
        let path = get_config_path();
        if path.exists() {
            let config = load_config(&path).expect("Failed to load config");

            // Verify FlowLM config
            assert_eq!(config.flow_lm.transformer.d_model, 1024);
            assert_eq!(config.flow_lm.transformer.num_heads, 16);
            assert_eq!(config.flow_lm.transformer.num_layers, 6);
            assert_eq!(config.flow_lm.flow.dim, 512);
            assert_eq!(config.flow_lm.flow.depth, 6);
            assert_eq!(config.flow_lm.lookup_table.n_bins, 4000);

            // Verify Mimi config
            assert_eq!(config.mimi.sample_rate, 24000);
            assert_eq!(config.mimi.channels, 1);
            assert!((config.mimi.frame_rate - 12.5).abs() < 1e-6);
            assert_eq!(config.mimi.seanet.dimension, 512);
            assert_eq!(config.mimi.seanet.ratios, vec![6, 5, 4]);
            assert_eq!(config.mimi.transformer.num_layers, 2);
            assert_eq!(config.mimi.quantizer.dimension, 32);
        } else {
            eprintln!("Config file not found at {:?}, skipping test", path);
        }
    }
}
