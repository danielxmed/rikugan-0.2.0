from __future__ import annotations

import numpy as np
import torch


SLICE_NAMES = ["resid_pre", "attn_out", "delta_attn", "mlp_out", "delta_mlp", "resid_post"]
NUM_SLICES = len(SLICE_NAMES)


def percentile_normalize(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Normalize array to [0, 1] using percentile clipping.

    Ignores the extreme 2% on each tail. Outliers are clamped to 0 or 1
    (still visible, not discarded). The middle 96% of values gets the
    full [0, 1] range regardless of distribution shape.

    Generalizes across:
    - Concentrated distributions (L2 norms of high-dim vectors)
    - Heavy-tailed distributions (outlier features in large models)
    - Normal distributions (behaves ~identically to z-score)
    """
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


class ActivationAccumulator:
    """Captures residual deltas from transformer layers via forward hooks."""

    def __init__(self, num_layers: int) -> None:
        self.residual_deltas: list[torch.Tensor | None] = [None] * num_layers
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            self.residual_deltas[layer_idx] = (out - inp).detach()
        return hook_fn

    def install(self, model: torch.nn.Module, adapter) -> None:
        modules = dict(model.named_modules())
        for i in range(len(self.residual_deltas)):
            path = adapter.module_path_for_layer(i)
            module = modules[path]
            handle = module.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def compute_block_heat(self) -> list[float]:
        norms: list[float] = []
        for delta in self.residual_deltas:
            if delta is not None:
                norm = delta.float().norm(dim=-1).mean().item()
                norms.append(norm)
            else:
                norms.append(0.0)
        return norms


class ActivationSliceAccumulator:
    """Captures per-layer activation slices (resid_pre, attn_out, mlp_out, resid_post)
    via forward hooks on layer, self_attn, and mlp modules."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        # Per-layer raw tensors: [seq_len, d_model] after squeeze
        self.resid_pre: list[torch.Tensor | None] = [None] * num_layers
        self.resid_post: list[torch.Tensor | None] = [None] * num_layers
        self.attn_out: list[torch.Tensor | None] = [None] * num_layers
        self.mlp_out: list[torch.Tensor | None] = [None] * num_layers
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_layer_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            self.resid_pre[layer_idx] = inp[0].detach()   # squeeze batch
            self.resid_post[layer_idx] = out[0].detach()
        return hook_fn

    def _make_attn_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.attn_out[layer_idx] = out[0].detach()
        return hook_fn

    def _make_mlp_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.mlp_out[layer_idx] = out[0].detach()
        return hook_fn

    def install(self, model: torch.nn.Module, adapter) -> None:
        modules = dict(model.named_modules())
        for i in range(self.num_layers):
            # Layer hook (resid_pre + resid_post)
            layer_path = adapter.module_path_for_layer(i)
            layer_mod = modules[layer_path]
            self._handles.append(layer_mod.register_forward_hook(self._make_layer_hook(i)))
            # Attention hook
            attn_path = adapter.module_path_for_attn(i)
            attn_mod = modules[attn_path]
            self._handles.append(attn_mod.register_forward_hook(self._make_attn_hook(i)))
            # MLP hook
            mlp_path = adapter.module_path_for_mlp(i)
            mlp_mod = modules[mlp_path]
            self._handles.append(mlp_mod.register_forward_hook(self._make_mlp_hook(i)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def compute_block_heat(self) -> list[float]:
        norms: list[float] = []
        for i in range(self.num_layers):
            pre = self.resid_pre[i]
            post = self.resid_post[i]
            if pre is not None and post is not None:
                delta = (post - pre).float()
                norm = delta.norm(dim=-1).mean().item()
                norms.append(norm)
            else:
                norms.append(0.0)
        return norms

    def normalize_and_serialize(self) -> tuple[bytes, dict, bytes, bytes]:
        """Percentile-normalize each slice per layer, clamp to [0,1], pack as float32 binary.

        Returns (slice_bytes, metadata_dict, token_proj_bytes, dim_proj_bytes).
        Binary layout: for layer L, slice S at index:
            offset = (L * 6 + S) * seq_len * d_model  (in float32 elements)
        Slice order: [resid_pre, attn_out, delta_attn, mlp_out, delta_mlp, resid_post]
        """
        # Get dimensions from first available tensor
        seq_len = 0
        d_model = 0
        for i in range(self.num_layers):
            if self.resid_pre[i] is not None:
                seq_len = self.resid_pre[i].shape[0]
                d_model = self.resid_pre[i].shape[1]
                break

        if seq_len == 0 or d_model == 0:
            return b"", {"slice_types": SLICE_NAMES, "num_layers": self.num_layers,
                         "seq_len": 0, "d_model": 0}, b"", b""

        total_floats = self.num_layers * NUM_SLICES * seq_len * d_model
        buf = np.zeros(total_floats, dtype=np.float32)

        zero = torch.zeros(seq_len, d_model)

        for layer_idx in range(self.num_layers):
            pre = self.resid_pre[layer_idx]
            post = self.resid_post[layer_idx]
            attn = self.attn_out[layer_idx]
            mlp = self.mlp_out[layer_idx]

            def to_cpu(t: torch.Tensor | None) -> torch.Tensor:
                return t.detach().float().cpu() if t is not None else zero

            pre_cpu = to_cpu(pre)
            attn_cpu = to_cpu(attn)
            mlp_cpu = to_cpu(mlp)
            post_cpu = to_cpu(post)

            # delta_attn = attn_out (attention sublayer output, not residual + attention)
            # delta_mlp = resid_post - resid_pre - attn_out (the MLP's actual contribution)
            delta_attn_cpu = attn_cpu
            delta_mlp_cpu = post_cpu - pre_cpu - attn_cpu

            slices_raw = [
                pre_cpu,           # resid_pre
                attn_cpu,          # attn_out
                delta_attn_cpu,    # delta_attn
                mlp_cpu,           # mlp_out
                delta_mlp_cpu,     # delta_mlp
                post_cpu,          # resid_post
            ]

            for slice_idx, tensor in enumerate(slices_raw):
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                normalized = percentile_normalize(arr)

                offset = (layer_idx * NUM_SLICES + slice_idx) * seq_len * d_model
                buf[offset:offset + seq_len * d_model] = normalized.ravel()

        # Compute lateral projections from pre-normalization raw tensors
        # Token projection: L2 norm over d_model axis -> [seq_len] per stage per layer
        token_proj_buf = np.zeros(self.num_layers * NUM_SLICES * seq_len, dtype=np.float32)
        # Dimension projection: L2 norm over seq_len axis -> [d_model] per stage per layer
        dim_proj_buf = np.zeros(self.num_layers * NUM_SLICES * d_model, dtype=np.float32)

        for layer_idx in range(self.num_layers):
            pre = self.resid_pre[layer_idx]
            post = self.resid_post[layer_idx]
            attn = self.attn_out[layer_idx]
            mlp = self.mlp_out[layer_idx]

            def to_cpu(t: torch.Tensor | None) -> torch.Tensor:
                return t.detach().float().cpu() if t is not None else zero

            pre_cpu = to_cpu(pre)
            attn_cpu = to_cpu(attn)
            mlp_cpu = to_cpu(mlp)
            post_cpu = to_cpu(post)
            delta_attn_cpu = attn_cpu
            delta_mlp_cpu = post_cpu - pre_cpu - attn_cpu

            raw_slices = [pre_cpu, attn_cpu, delta_attn_cpu, mlp_cpu, delta_mlp_cpu, post_cpu]

            for stage_idx, tensor in enumerate(raw_slices):
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor

                # Token projection: L2 norm over d_model -> [seq_len]
                tok_proj = np.linalg.norm(arr, axis=1)  # [seq_len]
                t_offset = layer_idx * NUM_SLICES * seq_len + stage_idx * seq_len
                token_proj_buf[t_offset:t_offset + seq_len] = percentile_normalize(tok_proj)

                # Dimension projection: L2 norm over seq_len -> [d_model]
                dim_proj = np.linalg.norm(arr, axis=0)  # [d_model]
                d_offset = layer_idx * NUM_SLICES * d_model + stage_idx * d_model
                dim_proj_buf[d_offset:d_offset + d_model] = percentile_normalize(dim_proj)

        meta = {
            "slice_types": SLICE_NAMES,
            "num_layers": self.num_layers,
            "seq_len": seq_len,
            "d_model": d_model,
        }
        return buf.tobytes(), meta, token_proj_buf.tobytes(), dim_proj_buf.tobytes()
