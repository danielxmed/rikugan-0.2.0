import * as api from '../../services/api';
import { useModelStore } from '../../stores/modelStore';

export async function loadModel(
  modelName: string,
  writeLine: (text: string) => void,
): Promise<void> {
  const store = useModelStore.getState();
  store.setLoading(modelName);
  writeLine(`Loading ${modelName}...`);

  try {
    const res = await api.loadModel(modelName);
    const info = await api.getModelInfo(res.adapter_id);
    useModelStore.getState().setModel(res.adapter_id, modelName, info);
    writeLine(`\x1b[32m${res.message}\x1b[0m`);
  } catch (e) {
    useModelStore.getState().setError();
    writeLine(`\x1b[31mError: ${e instanceof Error ? e.message : String(e)}\x1b[0m`);
  }
}

export async function getModelInfo(
  writeLine: (text: string) => void,
): Promise<void> {
  const store = useModelStore.getState();
  if (!store.modelId) {
    writeLine('\x1b[33mNo model loaded. Use: load <model-name>\x1b[0m');
    return;
  }
  try {
    const info = await api.getModelInfo(store.modelId);
    writeLine(`  Model:    ${info.display_name}`);
    writeLine(`  HF Repo:  ${info.hf_repo}`);
    writeLine(`  Params:   ${info.parameters_approx}`);
    writeLine(`  Layers:   ${info.layers}`);
    writeLine(`  Heads:    ${info.heads} (KV: ${info.kv_heads})`);
    writeLine(`  d_model:  ${info.d_model}`);
    writeLine(`  d_inter:  ${info.d_intermediate}`);
    writeLine(`  Vocab:    ${info.vocab_size}`);
  } catch (e) {
    writeLine(`\x1b[31mError: ${e instanceof Error ? e.message : String(e)}\x1b[0m`);
  }
}

export function showHelp(writeLine: (text: string) => void): void {
  writeLine('  \x1b[1mCommands:\x1b[0m');
  writeLine('  load <model>   Load a model (e.g. load qwen3-0.6b)');
  writeLine('  run <prompt>   Run inference on the loaded model');
  writeLine('  info           Show info about the loaded model');
  writeLine('  contrast <\u03B3>   Set power-law contrast (default 3.0, higher = sparser)');
  writeLine('  slice <name>   Set face slice (auto, resid_pre, attn_out, delta_attn, mlp_out, delta_mlp, resid_post)');
  writeLine('  help           Show this help message');
}
