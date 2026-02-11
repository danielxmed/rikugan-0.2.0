export interface WsInferenceRun {
  type: 'inference.run';
  payload: {
    prompt: string;
    max_new_tokens?: number;
  };
}

export interface WsInferenceResult {
  type: 'inference.result';
  payload: {
    text: string;
    model_id: string;
  };
}

export interface WsActivationFrame {
  type: 'activation.frame';
  payload: {
    block_heat: number[];
    model_id: string;
    prompt: string;
    num_layers: number;
  };
}

export interface WsActivationSlices {
  type: 'activation.slices';
  payload: {
    slice_types: string[];
    num_layers: number;
    seq_len: number;
    d_model: number;
  };
}

export interface WsActivationProjections {
  type: 'activation.projections';
  payload: {
    num_layers: number;
    seq_len: number;
    d_model: number;
    num_stages: number;
    token_proj_size: number;
  };
}

export interface WsError {
  type: 'error';
  payload: {
    message: string;
  };
}

export type WsOutgoing = WsInferenceRun;
export type WsIncoming = WsInferenceResult | WsActivationFrame | WsActivationSlices | WsActivationProjections | WsError;
