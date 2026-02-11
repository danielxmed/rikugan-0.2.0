# Rikugan — Interface Specification & Execution Roadmap v0.3

> This document defines the interface layout, interaction model, visual states, and phased execution plan for Rikugan. It assumes familiarity with v0.1 (concept) and v0.2 (architecture).

---

## MVP Target Model

**Qwen 3 0.6B** — the first model Rikugan must fully support.

| Property | Value |
|---|---|
| Parameters | ~0.6B |
| Layers | 28 |
| Attention heads | 16 (with GQA: 8 KV heads) |
| d_model | 1024 |
| d_mlp | 3072 (SwiGLU) |
| Vocab size | 151,936 |
| Context length | 32,768 (MVP target: 512 tokens) |
| Format | safetensors on HuggingFace |
| VRAM required | ~1.5 GB (fp16) |

Why this model: modern architecture (RoPE, SwiGLU, GQA), small enough to iterate fast on any GPU, large enough to exhibit real transformer behavior, and representative of the architecture patterns found in 7B–70B models. If Rikugan works with Qwen 0.6B, it works with anything.

---

## Interface Layout

The interface is a single full-screen application divided into four zones. No menus, no sidebars, no floating windows. The layout is fixed — only the inspection panel toggles visibility.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│                                                     │
│                                                     │
│                   Main Viewport                     │
│                  (R3F / Three.js)                    │
│                                                     │
│                                                     │
│                                          ┌────────┐ │
│                                          │Minimap │ │
│                                          └────────┘ │
├─────────────────────────────────────────────────────┤
│  Terminal (REPL)                                    │
│  > _                                                │
└─────────────────────────────────────────────────────┘
```

When inspection is active:

```
┌───────────────────────────────────────┬─────────────┐
│                                       │             │
│                                       │ Inspection  │
│                                       │   Panel     │
│            Main Viewport              │             │
│                                       │ - Value     │
│                                       │ - Histogram │
│                                       │ - Upstream  │
│                                  ┌──┐ │ - Downstream│
│                                  │mm│ │             │
│                                  └──┘ │             │
├───────────────────────────────────────┴─────────────┤
│  Terminal (REPL)                                    │
│  > _                                                │
└─────────────────────────────────────────────────────┘
```

### Zone 1: Main Viewport (70–80% of screen)

The 3D scene. Fills the upper portion of the screen. The researcher navigates with:

- **Orbit**: left-click drag (rotate around focal point)
- **Pan**: right-click drag or middle-click drag
- **Zoom**: scroll wheel (continuous zoom also drives level transitions)
- **Select**: click on a neuron, head, or edge to open inspection panel

The viewport background is dark (near-black) to maximize contrast with the cold-to-hot color palette.

### Zone 2: Minimap (corner overlay)

Fixed in the bottom-right corner of the viewport. Shows the full model at Level 1 (macro) at all times. A semi-transparent rectangle indicates the main camera's current frustum position. Clicking on the minimap teleports the main camera to that location.

Size: approximately 200×150px. Renders at 15 FPS independently from the main scene.

### Zone 3: Terminal (bottom strip)

Fixed-height strip at the bottom of the screen. Default height: 3 lines (expandable to ~10 lines via drag or hotkey). Powered by xterm.js, styled to match the dark aesthetic.

The terminal shows:

- Command prompt with blinking cursor
- Command output (model info, inference status, export confirmations)
- Progress indicators for long operations (model loading, inference)
- Error messages in red

Hotkey: **backtick (`)** toggles terminal focus. When focused, all keyboard input goes to the terminal. When unfocused, keyboard input maps to viewport shortcuts.

### Zone 4: Inspection Panel (slide-in, right side)

Hidden by default. Slides in from the right when the researcher selects a component (click in viewport or `inspect` command). Takes approximately 25% of screen width.

Contents vary by selected component type:

**For a neuron:**
- Layer and index identifier
- Raw activation value
- Z-score (normalized within layer)
- Histogram: activation distribution across all positions in the current sequence
- Top 5 upstream connections (by contribution magnitude)
- Top 5 downstream connections (by contribution magnitude)

**For an attention head:**
- Layer and head index
- Attention pattern matrix as a 2D heatmap (tokens × tokens)
- Mean attention entropy (how "focused" vs. "diffuse" this head is)
- Top attended-to positions for each query position

**For an edge (MLP contribution):**
- Source neuron and target layer
- Weight value
- Effective contribution: activation × weight magnitude
- Rank of this edge among all edges into the target

Close button or **Esc** dismisses the panel.

---

## Visual States

The application has five distinct visual states. The terminal prompt reflects the current state.

### State: Idle

No model loaded. Viewport shows an empty dark scene with a centered text prompt: `"load <model> to begin"`. Terminal is active and accepts `load` commands.

```
Terminal prompt: rikugan >
```

### State: Model Loaded

Model is in GPU memory. Viewport shows the model's architecture as a static gray structure (Level 1 macro view) — all blocks visible but uncolored (no activations yet). The researcher can navigate the structure and inspect architecture metadata, but there's no activation data to visualize.

```
Terminal prompt: rikugan [qwen3-0.6b] >
```

### State: Inference Running

A forward pass is in progress. Blocks light up progressively as each layer completes — the researcher sees the activation wave propagate through the model in real time. Terminal shows a progress indicator.

```
Terminal prompt: rikugan [qwen3-0.6b] running >
```

### State: Inference Complete (Frozen Frame)

The forward pass is complete. The full model is colored according to the final activation state. This is the primary exploration state — the researcher navigates freely across zoom levels, inspects components, and the visualization is static (frozen at the last timestep).

If recording is active, the researcher can enter playback mode from here.

```
Terminal prompt: rikugan [qwen3-0.6b] t=14 >
```

### State: Playback

Playing back a recorded inference sequence. The visualization updates frame by frame. Playback controls (play/pause/step/seek) are available via terminal or hotkeys. The current frame index is displayed in the terminal prompt.

```
Terminal prompt: rikugan [qwen3-0.6b] ▶ t=7/14 >
```

---

## Keyboard Shortcuts

These work when the terminal is not focused (viewport mode):

| Key | Action |
|---|---|
| `` ` `` (backtick) | Toggle terminal focus |
| `Space` | Play/Pause playback |
| `→` | Step forward one frame |
| `←` | Step backward one frame |
| `Home` | Rewind to first frame |
| `1` | Zoom to Level 1 (macro) |
| `2` | Zoom to Level 2 (meso) |
| `3` | Zoom to Level 3 (micro) |
| `Esc` | Close inspection panel / deselect |
| `F` | Focus camera on selected component |

---

## Execution Roadmap

Development is organized into six phases. Each phase produces a working, demonstrable artifact. No phase depends on future work — each one is a usable checkpoint.

### Phase 0 — Skeleton (Foundation)

**Goal**: Backend serves data, frontend renders a 3D scene, they talk to each other.

**Backend deliverables:**
- FastAPI project structure with uvicorn
- `/models` and `/models/{id}/load` REST endpoints (hardcoded for Qwen 0.6B initially)
- Model loading from safetensors via PyTorch
- WebSocket endpoint (`/ws`) that accepts connections and echoes messages
- Basic inference: run a forward pass on a prompt, return generated text via REST

**Frontend deliverables:**
- React + Vite project with R3F
- Empty 3D scene with orbit controls and dark background
- xterm.js terminal integrated, accepts text input
- WebSocket connection to backend (connect/disconnect)
- Zustand store skeleton (model state, viewport state, terminal state)

**Completion test**: Researcher opens browser, types `load qwen3-0.6b` in terminal, sees confirmation. Types `run "hello"`, gets generated text back in terminal. The 3D scene is empty but functional.

---

### Phase 1 — Static Macro View (The Model Lights Up)

**Goal**: Run inference and see the model colored by activation magnitude at Level 1.

**Backend deliverables:**
- Hook installation on all layers (`register_forward_hook`)
- Activation accumulator: captures post-attention, post-MLP, attention weights, residual deltas
- Level 1 processing: compute L2 norm of residual delta per layer
- Binary serialization: Level 1 data as Float32Array with batch header
- WebSocket streaming: send Level 1 data after forward pass

**Frontend deliverables:**
- Model geometry: 28 blocks (for Qwen 0.6B) arranged as 3D units along a vertical axis
- Color mapping: receive Level 1 floats, apply z-score normalization, γ power law, cold-to-hot palette
- Block coloring: update block materials based on activation magnitude
- Camera: top-down orbital view of the full model
- Contrast slider (γ) in terminal: `contrast <value>`

**Completion test**: Researcher runs `run "The capital of France is"`, sees 28 blocks light up with varying intensities. Hot blocks (high residual contribution) are red/white, passive blocks are deep blue. Changing γ with `contrast 4` makes the visualization sparser. Running a different prompt changes the heat pattern.

**This is the MVP moment.** If this works and looks compelling, the project is validated.

---

### Phase 2 — Meso View (Attention Arcs and MLP Heatmaps)

**Goal**: Zoom into a layer and see attention patterns and neuron activations.

**Backend deliverables:**
- Level 2 processing: z-score normalize attention patterns and MLP activations per layer
- On-demand data: backend sends Level 2 data for a specific layer when requested
- REST endpoint: `/inspect/head/{layer}/{head}` returns attention matrix
- REST endpoint: `/inspect/layer/{layer}/mlp` returns MLP activation heatmap

**Frontend deliverables:**
- Zoom transition: camera distance triggers Level 1 → Level 2 transition
- Attention arcs: render arcs between token positions, colored by attention weight
- MLP heatmap: render neuron activations as a textured plane within the MLP block
- Lazy loading: request Level 2 data from backend when entering a layer
- Token labels: display input tokens along the sequence axis
- `goto layer <L>` and `goto layer <L> attn/mlp` terminal commands

**Completion test**: Researcher runs inference, then types `goto layer 14 attn`. Camera smoothly moves into layer 14. Attention arcs appear between token positions. Typing `goto layer 14 mlp` shows the MLP neuron heatmap. Scrolling the mouse wheel zooms between macro and meso views fluidly.

---

### Phase 3 — Micro View (Neuron Navigation)

**Goal**: Navigate among individual neurons, see edges and contributions.

**Backend deliverables:**
- Level 3 processing: individual neuron z-scores and edge contribution computation
- Edge attribution for MLP: `contribution(neuron_i) = ‖activation_i × W_out[:, i]‖`
- On-demand data: send neuron-level data for a specific block

**Frontend deliverables:**
- Neuron rendering: InstancedMesh with per-instance color based on activation z-score
- Edge rendering: lines connecting neurons across adjacent layers, colored by contribution
- Navigation: WASD or drag to move through the neuron field
- Click interaction: click a neuron to select it, click an edge to see weight value
- Inspection panel: slides in with neuron/edge details

**Completion test**: From Level 2, researcher zooms into the MLP block. Individual neurons appear as colored points. Bright red neurons have high activation. Edges to downstream hot neurons glow. Clicking a neuron opens the inspection panel showing its activation value, z-score, and top connections.

---

### Phase 4 — Temporal System (Recording and Playback)

**Goal**: Record inference, play it back frame by frame.

**Backend deliverables:**
- Ring buffer implementation: store complete activation snapshots per timestep
- `generate` command: autoregressive generation that records each step
- Playback engine: `play(speed)`, `pause()`, `seek(t)`, `step(±1)`
- Frame streaming: send frames sequentially during playback, respecting backpressure

**Frontend deliverables:**
- Playback state management: current frame index, play/pause state, speed
- Frame rendering: update visualization on each received frame
- Playback controls: terminal commands (`play`, `pause`, `step`, `seek`, `rewind`)
- Keyboard shortcuts: Space (play/pause), arrow keys (step), Home (rewind)
- Frame indicator in terminal prompt: `▶ t=7/14`

**Completion test**: Researcher types `generate "Once upon a time" --max_tokens 20`. The model generates 20 tokens. Researcher types `rewind` then `play 0.25`. The visualization replays the generation at quarter speed — the researcher watches the activation landscape reconfigure token by token. Pressing Space pauses. Arrow keys step frame by frame.

---

### Phase 5 — Comparative Mode and Minimap

**Goal**: Compare activations between inputs. Full spatial awareness.

**Backend deliverables:**
- `compare` command: run two prompts, compute per-component activation differences
- Differential data: send delta activations (signed difference) for visualization

**Frontend deliverables:**
- Comparative visualization: diverging color palette (blue = decreased, red = increased)
- Side-by-side or overlay mode for differential maps
- Minimap: secondary R3F canvas rendering Level 1 at all times
- Minimap frustum indicator (wireframe rectangle showing main camera position)
- Minimap click-to-teleport
- `compare "<prompt_a>" "<prompt_b>"` terminal command

**Completion test**: Researcher types `compare "The doctor treated her" "The doctor treated his"`. The visualization shows a differential map — most of the model is dark (no change), but specific heads and neurons in certain layers light up red or blue, revealing gender-sensitive circuits. The minimap in the corner shows the full model with the researcher's viewport position.

---

### Phase 6 — Polish and Scale

**Goal**: Production-quality UX, support for larger models.

**Deliverables:**
- Tab completion in terminal for all commands, model names, and layer identifiers
- Command history (up/down arrows)
- `help <command>` inline documentation
- Export functionality: `export activations`, `export recording`, `export screenshot`
- Performance profiling and optimization for 7B-class models
- Loading indicators and progress bars for long operations
- Error handling and recovery (WebSocket reconnection, model load failures)
- Documentation: user guide with examples

**Completion test**: A researcher who has never seen Rikugan before can load a model, run inference, navigate all three zoom levels, record and play back a generation sequence, and compare two inputs — using only the terminal and mouse — within 15 minutes of first opening the tool.

---

## Phase Summary

| Phase | Name | Core Deliverable | Depends On |
|---|---|---|---|
| 0 | Skeleton | Backend + frontend talk to each other | — |
| 1 | Static Macro | **Model lights up after inference** (MVP) | Phase 0 |
| 2 | Meso View | Attention arcs + MLP heatmaps on zoom | Phase 1 |
| 3 | Micro View | Individual neuron navigation + edges | Phase 2 |
| 4 | Temporal | Recording + playback + slow motion | Phase 1 |
| 5 | Comparative | Differential activation maps + minimap | Phase 4 |
| 6 | Polish | UX, scale, export, documentation | Phase 3 + 5 |

Note: Phase 4 depends on Phase 1, not Phase 3. Temporal recording can be built on macro-level data before micro-level navigation exists. Phases 2–3 (zoom levels) and Phase 4 (temporal) can be developed in parallel.

---

*Document version 0.3 — Interface specification and execution roadmap.*
*Previous: v0.2 — Technical architecture. v0.1 — Conceptual foundation.*
