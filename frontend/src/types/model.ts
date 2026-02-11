export interface ModelInfo {
  adapter_id: string;
  display_name: string;
  hf_repo: string;
  aliases: string[];
  layers: number;
  heads: number;
  kv_heads: number;
  d_model: number;
  d_intermediate: number;
  vocab_size: number;
  max_seq_len: number;
  parameters_approx: string;
}

export interface ModelListEntry extends ModelInfo {
  loaded: boolean;
}

export interface LoadModelResponse {
  status: string;
  adapter_id: string;
  message: string;
}

export interface InferenceResponse {
  prompt: string;
  generated_text: string;
  model_id: string;
}
