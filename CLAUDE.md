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
Starts FastAPI on port 8080 with uvloop. Check for stale processes first: `ss -tlnp | grep 8080` (`lsof` is not available).

**Frontend**:
```bash
cd /workspace/rikugan-0.2.0/frontend && npm run dev
```
Starts Vite dev server on port 5173. Proxies `/api/*` → localhost:8080 (strips `/api` prefix), `/ws` → ws://localhost:8080/ws.

**Build & Lint**:
```bash
cd /workspace/rikugan-0.2.0/frontend && npm run build   # tsc + vite build
cd /workspace/rikugan-0.2.0/frontend && npm run lint     # eslint
```

No formal test suite exists. Verification is end-to-end via browser: load model → run inference → observe 3D visualization.

## Architecture

Client-server system: FastAPI backend on GPU VM, React frontend in browser. REST for discrete commands, WebSocket for streaming activation data.

### Backend (`backend/`)

- **Entry point**: `main.py` — FastAPI app with CORS, routes, adapter discovery on startup
- **Adapter pattern**: `adapters/__init__.py` defines `ModelAdapter` ABC + `AdapterRegistry` (auto-discovers adapters via pkgutil). New models = new adapter file in `adapters/`.
- **Qwen3 adapter**: `adapters/qwen3_0_6b.py` — loads with `attn_implementation="eager"` (required for forward hooks). Has `generate_with_hooks()` for Phase 1+.
- **Hooks**: `hooks.py` — `ActivationAccumulator` installs PyTorch forward hooks, captures residual deltas (output - input per layer), computes block_heat as L2 norms. Always `.detach()` tensors and `.float()` before `.norm()` (model runs bfloat16).
- **State**: `state.py` — `ServerState` singleton, one model loaded at a time
- **Config**: `config.py` — reads from `.env` (HF_TOKEN, DEVICE, DTYPE, etc.)
- **Routes**: `routes/models.py` (REST: list/load/info), `routes/inference.py` (REST fallback), `routes/ws.py` (WebSocket: `inference.run` → `activation.frame` + `inference.result`)

### Frontend (`frontend/src/`)

- **State**: Zustand stores in `stores/` — `modelStore` (model metadata/status), `activationStore` (block_heat Float32Array), `viewportStore` (zoom, gamma), `terminalStore` (history, focus)
- **3D Viewport**: `components/viewport/` — R3F Canvas → Scene → ModelBlocks. ModelBlocks uses InstancedMesh for all 28 layers in a single draw call. Color pipeline: z-score normalize → clamp to [0,1] → gamma power-law → 5-point palette (blue→red→white).
- **Terminal**: `components/terminal/` — xterm.js with manual input buffer tracking. `commandParser.ts` tokenizes/parses, `commandDispatcher.ts` routes, `commands.ts` implements handlers.
- **Services**: `services/websocket.ts` (singleton, auto-reconnect), `services/api.ts` (REST client with `/api` base)
- **Types**: `types/messages.ts` (WS protocol), `types/commands.ts` (ParsedCommand discriminated union), `types/model.ts` (model metadata)

### Communication Protocol

- **REST**: `POST /models/{name}/load`, `GET /models/{name}/info`, `POST /inference/run`
- **WebSocket** (`/ws`): Client sends `inference.run` → Server responds with `activation.frame` (block_heat array) + `inference.result` (generated text). Error responses as `error` type.

## Project Roadmap

Defined in `docs/project_bible/`. Current status:
- **Phase 0** (Skeleton): Complete — backend↔frontend communication
- **Phase 1** (Static Macro): Complete — model lights up with block heat after inference
- **Phase 2** (Meso View): Next — zoom into layers, attention arcs, MLP heatmaps
- **Phase 3** (Micro View): Navigate neurons, edges, contributions
- **Phase 4** (Temporal): Recording, playback, frame-by-frame
- **Phase 5** (Comparative): Differential activation maps
- **Phase 6** (Polish): Tab completion, help, export

## Key Constraints

- Always remove PyTorch hooks after use (memory leak risk)
- Use `@torch.inference_mode()` for hooked inference methods
- InstancedMesh count is fixed at creation — can't dynamically resize
- xterm.js has no "current line" API — input buffer is manually tracked
- Use narrow Zustand selectors to avoid unnecessary R3F re-renders
- Vite HMR uses WSS on port 443
