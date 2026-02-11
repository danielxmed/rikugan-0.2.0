# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python**: Use `/usr/bin/python3.11` — do NOT use `python3` (points to 3.14, incompatible)
- **GPU**: NVIDIA A40 with CUDA 12.8
- **Model**: Qwen/Qwen3-0.6B (28 layers, 16 heads, 8 KV heads, d_model=1024)

## Running the Application

**Backend** (must run from project root for module resolution):
```bash
cd /workspace/rikugan-0.2.0 && python3.11 -m backend.main
```
Starts FastAPI on port 8080 with uvloop, ws_max_size=400MB, ws_ping_timeout=60s. Check for stale processes first: `ss -tlnp | grep 8080` (`lsof` is not available).

**Frontend**:
```bash
cd /workspace/rikugan-0.2.0/frontend && npm run dev
```
Starts Vite dev server on port 5173. Proxies `/api/*` → localhost:8080 (strips `/api` prefix), `/ws` → ws://localhost:8080/ws. HMR uses WSS on port 443.

**Build & Lint**:
```bash
cd /workspace/rikugan-0.2.0/frontend && npm run build   # tsc + vite build
cd /workspace/rikugan-0.2.0/frontend && npm run lint     # eslint
```

No formal test suite exists. Verification is end-to-end via browser: load model → run inference → observe 3D visualization.

## Architecture

Client-server system: FastAPI backend on GPU VM, React frontend in browser. REST for discrete commands, WebSocket for streaming activation data including binary frames.

### Backend (`backend/`)

- **Entry point**: `main.py` — FastAPI app with CORS, routes, adapter discovery on startup, `/health` endpoint
- **Adapter pattern**: `adapters/__init__.py` defines `ModelAdapter` ABC + `AdapterRegistry` (auto-discovers via pkgutil). New models = new adapter file in `adapters/`.
- **ModelAdapter ABC** requires: `info()`, `load()`, `is_loaded()`, `unload()`, `generate()`, `module_path_for_layer()`, `module_path_for_attn()`, `module_path_for_mlp()`. Optional methods: `generate_with_hooks()` (Phase 1), `forward_with_slices()` (Phase 1.5+).
- **Qwen3 adapter**: `adapters/qwen3_0_6b.py` — loads with `attn_implementation="eager"` (required for forward hooks). Implements all three inference methods.
- **Hooks**: `hooks.py` — Two accumulator classes:
  - `ActivationAccumulator`: Residual deltas (output - input) per layer → block_heat L2 norms.
  - `ActivationSliceAccumulator`: Captures 6 slices per layer (`resid_pre`, `attn_out`, `delta_attn`, `mlp_out`, `delta_mlp`, `resid_post`) via hooks on layer, self_attn, and mlp modules. `normalize_and_serialize()` returns binary float32 arrays for slices + lateral projections (token/dimension L2 norms). Uses percentile normalization (p2–p98 clipping), not z-score.
- **State**: `state.py` — `ServerState` singleton, one model loaded at a time
- **Config**: `config.py` — reads from `.env`: `HOST`, `PORT`, `HF_TOKEN`, `MODEL_CACHE_DIR`, `DEVICE`, `DTYPE`, `MAX_SEQ_LEN`
- **Routes**: `routes/models.py` (REST: list/load/info), `routes/inference.py` (REST fallback), `routes/ws.py` (WebSocket with binary frame support)

### Frontend (`frontend/src/`)

- **State**: Zustand stores in `stores/`:
  - `modelStore` — model metadata/status
  - `activationStore` — block_heat, sliceData, sliceMeta, tokenProj, dimProj, projMeta
  - `viewportStore` — zoomLevel (1-3), focusedLayer, gamma, sliceDepth, boundaryLayer
  - `terminalStore` — history, focus
- **3D Viewport**: `components/viewport/` — R3F Canvas → Scene:
  - `ModelBlocks.tsx` — Per-layer meshes with 6 materials per block (top/bottom: full heatmap DataTextures `[d_model, seq_len]`; front/back: token projection bands; left/right: dimension projection bands). Falls back to InstancedMesh when no slice data.
  - `heatmapShader.ts` — Custom GLSL for palette mapping, gamma correction, band rendering, slice indicators
  - `blockLayout.ts` — Dynamic block dimensions based on seq_len and d_model
  - `SlicePlane.tsx` — Interactive coronal slice visualization
  - `BlockBoundary.tsx` — Camera collision detection
  - `ViewportHUD.tsx` — On-screen face legend, slice depth, color scale
  - `LayerLabels.tsx` — Layer index labels (L0, L1, …)
  - `useArrowPan.ts` — Arrow key panning
- **Terminal**: `components/terminal/` — xterm.js with manual input buffer tracking. `commandParser.ts` tokenizes/parses, `commandDispatcher.ts` routes, `commands.ts` implements handlers. Commands: `load`, `run`, `info`, `contrast`, `slice`, `help`.
- **Services**: `services/websocket.ts` (singleton, auto-reconnect, handles binary frames with tag dispatch), `services/api.ts` (REST client with `/api` base)
- **Types**: `types/messages.ts` (WS protocol), `types/commands.ts` (ParsedCommand discriminated union), `types/model.ts` (model metadata)

### Communication Protocol

- **REST**: `GET /health`, `GET /models`, `POST /models/{name}/load`, `GET /models/{name}/info`, `POST /inference/run`
- **WebSocket** (`/ws`): Client sends `inference.run` → Server responds with sequence:
  1. `activation.frame` (JSON) — block_heat array, backward compat
  2. `activation.slices` (JSON) — slice metadata (seq_len, d_model, slice_types)
  3. Binary frame: 4-byte tag `0x01` + float32 slice data (`num_layers * 6 * seq_len * d_model` floats)
  4. `activation.projections` (JSON) — projection metadata
  5. Binary frame: 4-byte tag `0x02` + token_proj + dim_proj float32 arrays
  6. `inference.result` (JSON) — generated text
- Falls back to `generate_with_hooks()` (no binary frames) or `generate()` (text only) if adapter lacks slice support.

## Project Roadmap

Defined in `docs/project_bible/`. Current status:
- **Phase 0** (Skeleton): Complete — backend↔frontend communication
- **Phase 1** (Static Macro): Complete — block heat after inference
- **Phase 1.5** (Volumetric Slices): Complete — 6 activation slices per layer, heatmap textures, lateral projections, slice plane
- **Phase 2** (Meso View): Next — zoom into layers, attention arcs, MLP heatmaps
- **Phase 3–6**: See roadmap doc

## Key Constraints

- Always remove PyTorch hooks after use (memory leak risk)
- Use `@torch.inference_mode()` for hooked inference methods
- Always `.detach()` tensors and `.float()` before `.norm()` (model runs bfloat16)
- xterm.js has no "current line" API — input buffer is manually tracked
- Use narrow Zustand selectors to avoid unnecessary R3F re-renders
- Backtick (`) toggles terminal focus in `App.tsx`
