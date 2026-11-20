#ifndef MBUFO_MODELS_QWEN3_TTS_CONFIG_HPP_
#define MBUFO_MODELS_QWEN3_TTS_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mbufo::models::qwen3_tts {

enum class ModelVariant : std::uint8_t {
  kBase,
  kVoiceDesign,
  kCustomVoice,
  kUnsupported,
};

/// Talker (main decoder) configuration. Mirrors `talker_config` in
/// Qwen3-TTS config.json.
struct TalkerConfig {
  std::uint32_t vocab_size = 3072;         // codec vocab (codec spans)
  std::uint32_t text_vocab_size = 151936;  // text LM vocab (spk/tts specials)
  std::uint32_t hidden_size = 2048;
  std::uint32_t intermediate_size = 6144;
  std::uint32_t num_hidden_layers = 28;
  std::uint32_t num_attention_heads = 16;
  std::uint32_t num_key_value_heads = 8;
  std::uint32_t head_dim = 128;
  std::uint32_t num_code_groups = 16;
  std::uint32_t text_hidden_size = 2048;
  std::uint32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-6F;
  float rope_theta = 1000000.0F;
  std::vector<std::uint32_t> mrope_section = {24, 20, 20};  // T,H,W chunks
  bool mrope_interleaved = true;

  // codec control ids
  std::uint32_t codec_eos_token_id = 2150;
  std::uint32_t codec_bos_id = 2149;
  std::uint32_t codec_pad_id = 2148;
  std::uint32_t codec_think_id = 2154;
  std::uint32_t codec_nothink_id = 2155;
  std::uint32_t codec_think_bos_id = 2156;
  std::uint32_t codec_think_eos_id = 2157;

  std::map<std::string, std::uint32_t> spk_id;
  std::map<std::string, std::uint32_t> codec_language_id;
  std::map<std::string, std::optional<std::string>> spk_dialect;
};

/// Code predictor (MTP sub-model) config.
/// `talker_config.code_predictor_config`.
struct CodePredictorConfig {
  std::uint32_t vocab_size = 2048;
  std::uint32_t hidden_size = 1024;
  std::uint32_t intermediate_size = 3072;
  std::uint32_t num_hidden_layers = 5;
  std::uint32_t num_attention_heads = 16;
  std::uint32_t num_key_value_heads = 8;
  std::uint32_t head_dim = 128;
  std::uint32_t num_code_groups = 16;
  std::uint32_t max_position_embeddings = 65536;
  float rms_norm_eps = 1e-6F;
  float rope_theta = 1000000.0F;
};

struct SpeechTokenizerConfig {
  std::string model_type;
  std::uint32_t input_sample_rate = 24000;
  std::uint32_t output_sample_rate = 24000;
  std::uint32_t decode_upsample_rate = 1920;
  std::uint32_t latent_dim = 1024;
  std::uint32_t codebook_dim = 512;
  std::uint32_t codebook_size = 2048;
  std::uint32_t decoder_dim = 1536;
  std::uint32_t hidden_size = 512;
  std::uint32_t intermediate_size = 1024;
  std::uint32_t head_dim = 64;
  std::uint32_t num_attention_heads = 16;
  std::uint32_t num_hidden_layers = 8;
  std::uint32_t num_key_value_heads = 16;
  std::uint32_t num_quantizers = 16;
  std::uint32_t num_semantic_quantizers = 1;
  std::uint32_t sliding_window = 72;
  float layer_scale_initial_scale = 0.01F;
  float rms_norm_eps = 1e-5F;
  float rope_theta = 10000.0F;
  std::vector<std::uint32_t> upsample_rates = {8, 5, 4, 3};
  std::vector<std::uint32_t> upsampling_ratios = {2, 2};

  std::uint32_t encode_downsample_rate = 1920;
  std::uint32_t encoder_valid_num_quantizers = 16;
  std::uint32_t encoder_audio_channels = 1;
  std::uint32_t encoder_codebook_dim = 256;
  std::uint32_t encoder_codebook_size = 2048;
  std::uint32_t encoder_hidden_size = 512;
  std::uint32_t encoder_intermediate_size = 2048;
  std::uint32_t encoder_head_dim = 64;
  std::uint32_t encoder_num_attention_heads = 8;
  std::uint32_t encoder_num_hidden_layers = 8;
  std::uint32_t encoder_num_key_value_heads = 8;
  std::uint32_t encoder_num_quantizers = 32;
  std::uint32_t encoder_num_semantic_quantizers = 1;
  std::uint32_t encoder_num_filters = 64;
  std::uint32_t encoder_num_residual_layers = 1;
  std::uint32_t encoder_kernel_size = 7;
  std::uint32_t encoder_last_kernel_size = 3;
  std::uint32_t encoder_residual_kernel_size = 3;
  std::uint32_t encoder_compress = 2;
  std::uint32_t encoder_dilation_growth_rate = 2;
  std::uint32_t encoder_sliding_window = 250;
  float encoder_layer_scale_initial_scale = 0.01F;
  float encoder_norm_eps = 1e-5F;
  float encoder_rope_theta = 10000.0F;
  std::vector<std::uint32_t> encoder_upsampling_ratios = {8, 6, 5, 4};
};

struct SpeakerEncoderConfig {
  std::uint32_t embedding_dim = 2048;
  std::uint32_t sample_rate = 24000;
};

/// Qwen3-TTS top-level model config.
struct ModelConfig {
  std::string model_type;
  std::string tokenizer_type;  // qwen3_tts_tokenizer_12hz
  std::string tts_model_size;  // 1b7
  std::string tts_model_type;  // custom_voice, voice_design, or base
  ModelVariant variant = ModelVariant::kUnsupported;

  std::uint32_t im_start_token_id = 151644;
  std::uint32_t im_end_token_id = 151645;
  std::uint32_t tts_pad_token_id = 151671;
  std::uint32_t tts_bos_token_id = 151672;
  std::uint32_t tts_eos_token_id = 151673;

  TalkerConfig talker;
  CodePredictorConfig code_predictor;
  SpeechTokenizerConfig speech_tokenizer;
  std::optional<SpeakerEncoderConfig> speaker_encoder;
};

[[nodiscard]] ModelVariant ParseModelVariant(std::string_view model_type);

/// This port intentionally accepts only the quality-qualified 1.7B 12 Hz
/// family. The 0.6B and experimental 25 Hz checkpoints are rejected.
[[nodiscard]] bool IsSupportedModelConfig(const ModelConfig& config);

/// Parses a Qwen3-TTS `config.json`.
[[nodiscard]] std::optional<ModelConfig> ParseModelConfig(
    const std::string& json_contents);

[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir);

[[nodiscard]] bool LoadSpeechTokenizerConfigFromPath(
    const std::string& model_dir, SpeechTokenizerConfig* config);

}  // namespace mbufo::models::qwen3_tts

#endif  // MBUFO_MODELS_QWEN3_TTS_CONFIG_HPP_
