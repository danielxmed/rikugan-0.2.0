# Rikugan — Technical Architecture v0.2

> This document describes the technical stack, data pipeline, wire protocol, and system architecture for Rikugan. It assumes familiarity with the conceptual foundation defined in v0.1.

---

## System Overview

Rikugan is a client-server system. The backend runs on a GPU-equipped VM and handles model loading, inference, activation capture, and data processing. The frontend runs in the researcher's browser and handles 3D visualization, user interaction, and the terminal REPL.

The two communicate over a combination of REST (for discrete commands and metadata) and binary WebSocket (for streaming activation data during inference).

```
┌─────────────────────────────────────┐
│          Researcher's Browser       │
│                                     │
│  ┌───────────┐    ┌──────────────┐  │
│  │ Web Worker │───▶│  Main Thread │  │
│  │ (WebSocket │    │  (React+R3F) │  │
│  │  receiver) │    │              │  │
│  └───────────┘    │  ┌─────────┐ │  │
│   Transferable    │  │ Three.js│ │  │
│    Objects ──────▶│  │  WebGL  │ │  │
│                   │  └─────────┘ │  │
│                   │  ┌─────────┐ │  │
│                   │  │Terminal │ │  │
│                   │  │ (REPL)  │ │  │
│                   │  └─────────┘ │  │
│                   └──────────────┘  │
└──────────────┬──────────────────────┘
               │ WS (binary) + REST (JSON)
               │
┌──────────────▼──────────────────────┐
│        GPU VM (A100 / similar)      │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  FastAPI + uvicorn (uvloop)  │   │
│  │                              │   │
│  │  ┌────────┐  ┌────────────┐  │   │
│  │  │PyTorch │  │  Activation │  │   │
│  │  │Inference│─▶│  Processor  │  │   │
│  │  │+ Hooks │  │(z-score,agg)│  │   │
│  │  └────────┘  └─────┬──────┘  │   │
│  │                    │         │   │
│  │  ┌────────────────▼──────┐   │   │
│  │  │   Temporal Recorder   │   │   │
│  │  │    (Ring Buffer)      │   │   │
│  │  └───────────────────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Deployment Model

The backend runs on a remote VM with a high-end GPU (A100 80GB as baseline). The frontend is a static site — it can be served from anywhere (CDN, local file, the VM itself). The researcher connects by pointing the frontend to the backend's address:

```
rikugan.connect("ws://203.0.113.42:8080")
```

This separation means the frontend has zero hardware requirements beyond a modern browser with WebGL support. All computation happens server-side.

---

## Backend Architecture

### Stack

- **Framework**: FastAPI (Starlette) with uvicorn
- **Runtime**: Python 3.11+
- **ML**: PyTorch 2.x with safetensors for model loading
- **GPU**: CUDA-capable GPU with sufficient VRAM for the target model

### uvicorn Configuration

```
uvicorn main:app \
  --ws-max-size 209715200 \       # 200MB max WebSocket message
  --ws-per-message-deflate false \ # no compression (dense float data)
  --loop uvloop \                  # ~25% faster async loop
  --ws-ping-timeout 60             # long timeout for large transfers
```

Rationale: Float32 arrays are dense numerical data that compresses poorly. Per-message-deflate adds CPU cost and per-connection memory overhead for negligible size reduction. Disabling it is a net win.

### API Surface

**REST endpoints** (JSON) — for discrete operations:

| Endpoint | Method | Purpose |
|---|---|---|
| `/models` | GET | List available models |
| `/models/{id}/load` | POST | Load a model into GPU memory |
| `/models/{id}/info` | GET | Model metadata: layers, heads, d_model, vocab size |
| `/models/{id}/layers` | GET | Detailed layer structure with shapes |
| `/inference/run` | POST | Start a forward pass with a given prompt |
| `/inference/state` | GET | Current inference state (idle, running, complete) |
| `/recording/frames` | GET | List recorded frames with timestamps |
| `/recording/frame/{t}` | GET | Retrieve a specific frame's metadata |

**WebSocket endpoint** (`/ws`) — for bidirectional streaming:

- **Server → Client**: activation data (binary chunks) during inference and playback
- **Client → Server**: control commands from the terminal REPL, ACK messages for backpressure

The WebSocket connection is persistent for the duration of a session. REST is used for stateless queries; WebSocket is used for anything that involves streaming or real-time state changes.

### Model Loading and Inference

Models are loaded from safetensors files using the `safetensors` library with PyTorch integration. The loading is lazy — tensor data is memory-mapped and only fully loaded when accessed.

Inference uses PyTorch's native `register_forward_hook` mechanism to capture activations at each layer. Hooks are installed on every relevant module:

- Post-attention activations (after the attention sublayer + residual add)
- Post-MLP activations (after the MLP sublayer + residual add)
- Attention weight matrices (the softmax output for each head)
- Residual deltas (the difference each sublayer adds to the stream)

Each hook writes its captured tensor to a shared accumulator. When the forward pass completes, the accumulator contains a complete activation snapshot for that timestep.

### Activation Processing Pipeline

Raw activations from hooks go through a processing pipeline before being sent to the frontend. The pipeline is zoom-level-aware — it produces different outputs depending on what the frontend is currently displaying.

**For Level 1 (Macro):**

One scalar per layer — the L2 norm of the residual added by that layer:

```
block_heat(layer_k) = ‖residual_added_by_layer_k‖₂
```

This is an array of N floats (where N = number of layers). Tiny payload, sent immediately.

**For Level 2 (Meso):**

Per-head attention patterns and per-neuron MLP activations for a specific layer. Attention patterns are shaped [num_heads, seq_len, seq_len]. MLP activations are shaped [seq_len, d_mlp]. Both are z-score normalized within the layer before sending:

```
normalized(x) = (x - μ_layer) / σ_layer
```

**For Level 3 (Micro):**

Individual neuron activations and edge contributions within a specific block. Neuron activations are the z-score normalized values. Edge contributions for MLP neurons are computed as:

```
contribution(neuron_i) = ‖activation_i × W_out[:, i]‖
```

This is computed on-demand when the researcher enters micro view for a specific block.

**The backend always computes Level 1 data for every forward pass** (it's cheap — just L2 norms). Levels 2 and 3 are computed on-demand when the frontend requests them for a specific layer or block.

### Tensor Serialization

The path from PyTorch tensor to wire bytes:

```python
# For GPU tensors: async copy to CPU
cpu_tensor = tensor.detach().to('cpu', non_blocking=True)
torch.cuda.synchronize()  # wait for transfer

# Serialize to bytes
raw_bytes = cpu_tensor.numpy().tobytes()
```

Notes:
- `.numpy()` is zero-copy for CPU contiguous tensors without gradients
- `.detach()` removes the tensor from the computation graph
- `non_blocking=True` allows overlapping GPU→CPU transfer with other work
- The resulting bytes are a flat array of float32 values in C-contiguous order

### Temporal Recording (Ring Buffer)

The backend maintains a ring buffer of activation snapshots in memory. Each snapshot contains the complete model state at one timestep of inference.

**Snapshot contents:**

- All layer activations (post-attention, post-MLP) as raw tensors
- Attention weight matrices for every head
- Residual deltas added by each layer
- Output logits and top-k token probabilities
- Metadata: timestep index, generated token, cumulative sequence

**Memory budget (GPT-2 small, 12 layers, 12 heads, d_model=768, seq_len=128):**

| Component | Shape | Size per layer | Total |
|---|---|---|---|
| Post-attention acts | [128, 768] | 384 KB | 4.6 MB |
| Post-MLP acts | [128, 768] | 384 KB | 4.6 MB |
| Attention weights | [12, 128, 128] | 768 KB | 9.2 MB |
| Residual deltas | [128, 768] | 384 KB | 4.6 MB |
| Logits (1 layer) | [128, 50257] | 24.5 MB | 24.5 MB |
| **Total per frame** | | | **~48 MB** |

For 200 frames (tokens), that is ~10 GB. Fits comfortably in a 64GB+ VM. For 7B parameter models, frames are larger (~500 MB each), but with 256GB RAM VMs available on cloud providers, 100+ frames remain feasible.

No compression, no delta encoding. Raw tensors in memory. Simplicity over engineering cleverness — the hardware budget absorbs it.

**Playback interface:**

The ring buffer exposes temporal navigation to the frontend:

- `play(speed)` — stream frames sequentially at adjustable speed
- `pause()` — stop at current frame
- `seek(t)` — jump to frame t
- `step(±1)` — advance or rewind one frame
- `get_frame(t)` — return a specific frame's data (processed for current zoom level)

---

## Wire Protocol

### Binary Frame Format

All activation data travels as binary WebSocket messages. Each message is chunked at the application level (512 KB per chunk) with a 16-byte binary header per chunk:

```
┌──────────────────────────────────────────────┐
│  Chunk Header (16 bytes, little-endian)       │
│  ┌──────────┬──────────┬──────────┬─────────┐ │
│  │ batch_id │ chunk_idx│ total    │ offset  │ │
│  │ uint32   │ uint32   │ uint32   │ uint32  │ │
│  └──────────┴──────────┴──────────┴─────────┘ │
│  Chunk Payload (up to 512 KB)                 │
│  [raw bytes...]                               │
└──────────────────────────────────────────────┘
```

The first chunk of each batch carries an additional **batch header** — a JSON-encoded metadata block prepended to the payload:

```json
{
  "model": "gpt2-small",
  "hook": "blocks.4.hook_mlp_out",
  "hook_hf": "transformer.h.4.mlp",
  "dtype": "float32",
  "shape": [128, 3072],
  "timestep": 7,
  "seq_position": 14,
  "zoom_level": 2,
  "normalization": "zscore_intra_layer"
}
```

The batch header is delimited from the binary payload by a 4-byte length prefix (uint32) at the start of the first chunk's payload, indicating how many bytes of JSON follow before the raw float data begins.

### Naming Convention

Hook point names follow **TransformerLens conventions** as primary identifiers, with **HuggingFace-native module paths** as secondary:

| TransformerLens (primary) | HuggingFace (secondary) | Description |
|---|---|---|
| `blocks.{L}.hook_resid_pre` | `model.layers.{L}` (input) | Residual stream before layer L |
| `blocks.{L}.hook_resid_post` | `model.layers.{L}` (output) | Residual stream after layer L |
| `blocks.{L}.attn.hook_pattern` | `model.layers.{L}.self_attn` | Attention patterns [heads, seq, seq] |
| `blocks.{L}.hook_attn_out` | `model.layers.{L}.self_attn` (output) | Attention sublayer output |
| `blocks.{L}.hook_mlp_out` | `model.layers.{L}.mlp` (output) | MLP sublayer output |
| `blocks.{L}.hook_resid_delta_attn` | — | Residual added by attention |
| `blocks.{L}.hook_resid_delta_mlp` | — | Residual added by MLP |

Both names are included in the batch header metadata. The frontend uses TransformerLens names for display and navigation; the HuggingFace names are stored for compatibility with external tools (nnsight, SAELens).

### Backpressure Protocol

The browser WebSocket API provides no native backpressure mechanism. Rikugan implements application-level flow control:

1. Server sends a chunk
2. Server waits for ACK from client (or timeout after configurable interval)
3. Client's Web Worker sends ACK after processing each chunk
4. Server sends next chunk upon receiving ACK

ACK messages are minimal: 4 bytes containing the batch_id being acknowledged.

This prevents the server from overwhelming the browser's memory when inference produces data faster than the frontend can render it.

---

## Frontend Architecture

### Stack

- **UI Framework**: React 18+
- **3D Rendering**: React Three Fiber (R3F) with Three.js
- **Terminal**: xterm.js (embedded, connected to Rikugan REPL — not a system shell)
- **State Management**: Zustand (lightweight, works well with R3F's render loop)
- **Build**: Vite

### Thread Architecture

The frontend runs on two threads:

**Web Worker thread**: owns the WebSocket connection. Receives binary chunks, reassembles them into complete Float32Array batches using pre-allocated buffers, then transfers completed batches to the main thread via Transferable Objects (~6ms for 32MB, zero-copy).

**Main thread**: React rendering, R3F/Three.js scene management, user interaction, terminal REPL. Receives ready-to-render Float32Array data from the worker and uploads it to WebGL buffers.

```
Web Worker                          Main Thread
──────────                          ───────────
ws.onmessage(chunk)
  → reassemble into buffer
  → postMessage(buffer, [buffer])  → onmessage(buffer)
     (Transferable, zero-copy)        → Float32Array view
                                      → bufferSubData() to GPU
                                      → R3F re-render
```

Critical configuration: `ws.binaryType = "arraybuffer"` must be set on the WebSocket. The default `"blob"` type causes the browser to spool data to disk, requiring an additional async read. ArrayBuffer mode gives direct access to bytes.

### WebGL Rendering Patterns

**Neuron rendering (Level 3):** Neurons are rendered via `InstancedMesh` — a single draw call for thousands of neurons, each with its own position and color. The color attribute buffer is pre-allocated with `DynamicDrawUsage` and updated per-frame via `bufferSubData()` with an `updateRange` specifying only the changed portion. This minimizes GPU upload cost.

**Attention arcs (Level 2):** Rendered as instanced lines or curved tubes connecting token positions. Thickness and color are driven by attention weight magnitude. For heads with very sparse attention (most weights near zero), only weights above a threshold are rendered — reducing visual clutter and draw calls.

**Heatmaps (Level 2, MLP):** Rendered as a textured plane where the texture data is a Float32Array of neuron activations mapped to the color palette. Texture updates use `texSubImage2D()` for partial updates.

**Level transitions:** Lazy loading is driven by camera distance. As the camera crosses a threshold (defined per level), the frontend requests the appropriate data granularity from the backend. Level 1 data is always resident in memory. Level 2 data is requested per-layer when the researcher zooms in. Level 3 data is requested per-block.

### Minimap

A secondary R3F canvas in the corner of the screen renders the full model at Level 1 (macro view) at all times. A wireframe rectangle indicates the main camera's current viewport position and orientation. Clicking on the minimap teleports the main camera to that location.

The minimap uses its own camera and renderer instance, running at a lower update frequency (15 FPS) to minimize overhead.

### Inspection Panel

A React component (not 3D) that slides in from the side when a neuron, head, or edge is selected. It receives data from the backend via REST (`/inspect/neuron/{layer}/{index}`, `/inspect/head/{layer}/{head}`, etc.) and displays:

- Raw activation value and z-score
- Histogram of activations across the current sequence (rendered with a lightweight chart library or canvas)
- Upstream/downstream connections ranked by contribution magnitude
- For attention heads: the full attention pattern matrix as a 2D heatmap

---

## Terminal REPL

The terminal is not a system shell. It is a custom REPL that parses a Rikugan-specific command language and dispatches commands to the backend (via REST or WebSocket) or to the frontend's visualization state directly.

### Command Language

**Model management:**

| Command | Action | Dispatch |
|---|---|---|
| `load <model_name>` | Load model into GPU memory | REST POST |
| `unload` | Release model from memory | REST POST |
| `info` | Display current model metadata | REST GET |

**Inference:**

| Command | Action | Dispatch |
|---|---|---|
| `run "<prompt>"` | Execute forward pass, populate visualization | WS |
| `generate "<prompt>" [--max_tokens N]` | Autoregressive generation with recording | WS |
| `compare "<prompt_a>" "<prompt_b>"` | Differential activation map | WS |

**Navigation:**

| Command | Action | Dispatch |
|---|---|---|
| `goto layer <L>` | Move camera to layer L (Level 2) | Frontend |
| `goto layer <L> mlp` | Move camera to MLP block of layer L | Frontend |
| `goto layer <L> attn` | Move camera to attention block of layer L | Frontend |
| `zoom macro` | Pull camera to Level 1 | Frontend |
| `zoom meso` | Set camera to Level 2 at current position | Frontend |
| `zoom micro` | Push camera to Level 3 at current position | Frontend |

**Inspection:**

| Command | Action | Dispatch |
|---|---|---|
| `inspect neuron <L> <index>` | Open inspection panel for a neuron | REST GET + Frontend |
| `inspect head <L>.<H>` | Open inspection panel for an attention head | REST GET + Frontend |
| `highlight head <L>.<H>` | Visually highlight a specific head | Frontend |
| `highlight layer <L>` | Visually highlight a specific layer | Frontend |

**Playback:**

| Command | Action | Dispatch |
|---|---|---|
| `play [speed]` | Play recorded frames (default 1x) | WS |
| `pause` | Freeze at current frame | WS |
| `step [+N\|-N]` | Step forward or backward N frames | WS |
| `seek <frame>` | Jump to specific frame | WS |
| `rewind` | Go to first frame | WS |

**Visualization:**

| Command | Action | Dispatch |
|---|---|---|
| `contrast <γ>` | Set the power-law contrast value | Frontend |
| `palette <name>` | Switch color palette | Frontend |
| `threshold <value>` | Hide activations below threshold | Frontend |
| `cut coronal <L>` | Show coronal cross-section at layer L | Frontend |

**Export:**

| Command | Action | Dispatch |
|---|---|---|
| `export activations <L_start>-<L_end>` | Save activation data to file | REST GET |
| `export recording [format]` | Save full temporal recording | REST GET |
| `export screenshot` | Capture current viewport | Frontend |

### REPL Architecture

The terminal is an xterm.js instance that captures keystrokes and feeds them to a **parser running in the main thread**. The parser tokenizes the command, validates syntax, and routes to the appropriate handler:

- **Frontend commands** (navigation, visualization settings) are executed immediately by updating React/Zustand state
- **Backend commands** (load, run, export) are sent to the server via REST or WebSocket and the terminal displays progress/results as they arrive
- **Hybrid commands** (inspect, highlight) trigger both a backend data request and a frontend visual update

The terminal supports command history (up/down arrows), tab completion for known commands and model names, and inline help (`help <command>`).

---

## Data Flow Summary

### Forward Pass (complete pipeline)

```
1. Researcher types: run "The capital of France is"
2. Terminal REPL parses command, sends to backend via WebSocket
3. Backend tokenizes prompt, runs forward pass with hooks active
4. Each hook captures its tensor and writes to accumulator
5. Forward pass completes. Accumulator holds full activation snapshot.
6. Snapshot is written to ring buffer (frame T)
7. Backend processes snapshot for current zoom level:
   - Level 1: compute L2 norms per layer → [N_layers] floats
   - Level 2: z-score normalize requested layer → send on demand
   - Level 3: compute neuron acts + edge contributions → send on demand
8. Processed data is serialized: JSON header + raw float32 bytes
9. Serialized data is chunked (512 KB per chunk, 16-byte chunk header)
10. Chunks are sent via WebSocket with ACK backpressure
11. Web Worker receives chunks, reassembles into Float32Array buffer
12. Worker transfers buffer to main thread (Transferable Object, ~6ms)
13. Main thread uploads to WebGL (bufferSubData on pre-allocated buffer)
14. R3F re-renders scene with new activation colors
15. Researcher sees the model light up.
```

### Playback (temporal navigation)

```
1. Researcher types: play 0.5
2. Terminal sends playback command to backend via WebSocket
3. Backend begins reading from ring buffer at half speed
4. For each frame: process for current zoom level, chunk, send
5. Frontend receives and renders each frame sequentially
6. Researcher sees the forward pass unfold in slow motion
7. At any point: pause, step, seek, rewind — all via terminal or hotkeys
```

### Zoom Transition

```
1. Researcher scrolls mouse wheel (or types: zoom meso)
2. Camera distance crosses Level 2 threshold
3. Frontend requests Level 2 data for the nearest layer from backend
4. Backend processes that layer's activation snapshot (z-score, etc.)
5. Data arrives, Three.js renders attention arcs and MLP heatmap
6. Level 1 blocks are still visible in the background (fade out with distance)
7. Minimap updates to show current viewport position
```

---

## Hardware Assumptions

### Minimum Recommended (target: GPT-2 class, ~124M params)

- **GPU**: Any CUDA GPU with 8GB+ VRAM
- **RAM**: 32 GB
- **Storage**: 10 GB free (model weights + temporary recordings)

### Recommended (target: 7B class models)

- **GPU**: A100 80GB or equivalent
- **RAM**: 64–128 GB
- **Storage**: 50 GB free

### High-end (target: 13B+ models, long recordings)

- **GPU**: A100 80GB
- **RAM**: 256 GB
- **Storage**: 100 GB+ free (for exported recordings)

The frontend requires only a modern browser with WebGL 2.0 support. No GPU required on the client side.

---

*Document version 0.2 — Technical architecture, data pipeline, and wire protocol.*
*Previous: v0.1 — Conceptual foundation and visual-mathematical framework.*
*Next: v0.3 — Interface specification and execution roadmap.*
