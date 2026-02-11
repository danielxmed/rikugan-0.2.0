# Rikugan — Concept Document v0.1

> *"To use the Limitless technique, one requires the Six Eyes (Rikugan). It allows for the precise manipulation of cursed energy at an atomic level, processing immense amounts of visual information to govern space itself without energy loss."*

---

## What is Rikugan?

Rikugan is a real-time, interactive 3D visualization tool for transformer-based language models. It allows researchers to inspect a model's internal representations — from macro architecture down to individual neurons — as inference happens, frame by frame.

The core idea: load a model (via safetensors or any standard format), run inference, and **watch the network think**. Activations light up. Circuits emerge. Patterns become visible across layers, heads, and neurons.

It is, in essence, a **microscope for transformers** — built for the mechanistic interpretability community, but accessible enough that anyone curious about what happens inside a language model can explore it.

---

## The Residual Stream as Central Metaphor

Rikugan does not represent transformers as "stacked layers" (the standard diagram). Instead, it adopts the **residual stream** framework from mechanistic interpretability research.

The model is visualized as a **river of information** flowing from the input embedding to the final logit output. Each layer (attention + MLP) is a **station** along the river: it reads from the stream, processes information, and writes a residual back. The residual stream itself — the sum of all contributions — is the backbone of the visualization.

Each token in the input sequence has its own parallel channel within the river. For a 128-token input, 128 streams flow side by side. The visual narrative follows these streams as they are progressively shaped by each layer.

---

## Visualization Levels

Rikugan operates across three spatial zoom levels plus an inspection mode. The transition between levels uses lazy loading: as the researcher zooms in, the system loads and renders progressively finer-grained data.

### Level 1 — Macro (Orbital View)

The researcher sees the full model from above, like a satellite view or a full-body MRI scan.

Each transformer block is rendered as a discrete 3D unit. Color and brightness encode the **magnitude of that block's contribution to the residual stream**, measured as the L2 norm of the residual added by that layer:

```
block_heat(layer_k) = ‖residual_added_by_layer_k‖₂
```

This gives an instant "thermal map" of the model: which layers are doing the heavy lifting for this specific input, and which are nearly passive. Early layers (typically syntactic processing) may light up differently from middle layers (semantic) and late layers (prediction).

The macro view supports multiple camera angles — top-down, lateral, anterior-posterior — and also **internal coronal cuts**: the researcher can slice through the model at any depth to see a cross-section of activation magnitudes at that point in the architecture.

### Level 2 — Meso (Coronal / Cross-Section View)

The researcher enters a specific timestep or layer and sees a cross-sectional "slice" of the model's state at that moment.

At this level, individual components become visible:

**Attention heads** appear as sub-blocks within the attention layer. Each head's activity is represented by arcs connecting token channels — the attention pattern. Arc thickness and color encode attention weight magnitude. This makes phenomena like induction heads, positional attention, and long-range dependencies immediately visible.

**MLP neurons** appear as a heatmap within the MLP sub-block. Each neuron's color encodes its activation magnitude, normalized within the layer (see the mathematical framework below). Hot neurons are potential feature detectors.

This level is where the researcher can identify co-activation patterns: which regions of the model are "hot" simultaneously during a given processing step, suggesting functional circuits.

### Level 3 — Micro (Navigable Map)

The researcher is now inside a specific block, navigating among individual neurons as if exploring a terrain map.

Neurons are rendered as discrete points in space. Their brightness and color encode activation values (z-score normalized). The researcher navigates spatially and can look "backward" along the temporal axis to see connections (edges) linking neurons across adjacent layers.

Edges connecting to high-activation neurons in the downstream layer are rendered brighter and warmer. Clicking an edge reveals the underlying weight value. This creates a natural visual flow: circuits emerge as chains of bright nodes connected by bright edges.

### Inspection Mode (not a zoom level)

At any zoom level, the researcher can select a specific neuron, head, or edge to open an inspection panel. This panel shows:

- Activation value (raw and z-score normalized)
- Activation histogram across the current sequence
- Activation histogram across multiple inputs (if recorded)
- Upstream and downstream connections ranked by contribution
- For attention heads: full attention pattern matrix for that head

Inspection mode is contextual — it enriches whatever zoom level the researcher is currently in, rather than requiring navigation to a separate view.

---

## Mathematical Framework

### Activation Coloring

All activation-based coloring uses **z-score normalization within each layer** to ensure visual consistency across layers with different magnitude scales:

```
intensity(neuron_i, layer_k) = (activation_i - μ_k) / σ_k
```

Where μ_k and σ_k are the mean and standard deviation of all neuron activations within layer k at the current timestep.

This ensures that a neuron is colored based on how active it is **relative to its peers in the same layer**, not in absolute terms. Without this, later layers (which tend to have larger absolute activations) would always appear hotter, which is misleading.

### Contrast Scaling (γ Power Law)

Raw z-scores produce a visually "noisy" heatmap where everything looks moderately warm. To make true signal stand out, Rikugan applies a power-law contrast function:

```
display_intensity(x) = colormap(x^γ)    where γ > 1
```

With γ = 3 or 4, only genuinely high activations reach warm colors; moderate activations are pushed toward cool tones. This produces the desired effect: circuits appear as rivers of red against a dark blue background.

The γ value is exposed as a **contrast slider** in the interface, allowing the researcher to tune sparsity of the visualization in real time.

### Color Palette

Cold-to-hot gradient: deep blue (inactive / below mean) → red (highly active) → bright white-red (extreme outliers).

### Edge Attribution

Since weights are static (they do not change during inference), edges between layers are colored based on their **effective contribution** to downstream activations:

**For attention**: edge color directly encodes the attention weight between two token positions in a given head. This is a direct output of the softmax and requires no additional computation.

**For MLP**: each neuron's contribution to the residual stream is:

```
contribution(neuron_i) = ‖activation_i × W_out[:, i]‖
```

Where activation_i is the post-nonlinearity scalar and W_out[:, i] is the corresponding column of the output projection matrix. This gives the magnitude of that neuron's "push" on the residual stream.

Edges from high-contribution neurons are rendered hotter. This allows the researcher to trace backward from a hot downstream neuron and identify which upstream connections were responsible.

---

## Temporal System

### Recording and Playback

Rikugan captures a complete **activation snapshot** at each step of the forward pass. A snapshot includes:

- All layer activations (post-attention, post-MLP)
- Attention weight matrices for every head
- Residual deltas added by each layer
- Output logits and top-k token probabilities

These snapshots are stored as a temporal sequence (the "recording"). The researcher can:

- **Play**: watch the forward pass unfold in real time or at adjustable speed
- **Pause**: freeze at any timestep and inspect the full model state
- **Rewind**: step backward frame by frame
- **Slow motion**: advance one token-generation step at a time, observing how the activation landscape reconfigures

### Temporal Navigation via Coronal Cuts

The coronal cut metaphor applies temporally as well as spatially. A "temporal coronal cut" shows the model's complete activation state at a single processing step — analogous to freezing time and taking a cross-sectional scan. The researcher can scrub through these cuts like frames in a video.

### Comparative Mode

The researcher can run two (or more) inputs and see **differential activation maps**: which neurons and heads change their behavior between inputs. This is the visual equivalent of causal intervention studies in mechanistic interpretability.

Example use case: comparing `"The doctor treated her patient"` vs `"The doctor treated his patient"` to identify gender-sensitive circuits.

---

## Interface Layout

### Main Viewport

The central area of the screen is the 3D scene: the model visualization rendered via WebGL. The researcher navigates with standard 3D controls (orbit, pan, zoom) and transitions between zoom levels fluidly.

### Minimap

A small panel in the corner of the screen shows the full model at Level 1 (macro) at all times, with a rectangle indicating the researcher's current viewport position. This functions like a minimap in open-world games: it prevents disorientation when deep in micro-level navigation.

### Terminal

An embedded terminal at the bottom of the screen serves as the primary control interface. The researcher issues commands to:

- Load models: `load gpt2-small`
- Run inference: `run "The capital of France is"`
- Navigate: `goto layer 10 mlp`
- Highlight components: `highlight head 7.3`
- Compare inputs: `compare "he is a" "she is a"`
- Control playback: `play`, `pause`, `rewind`, `step`
- Probe features: `probe layer 10 mlp neuron 423`
- Export data: `export activations layer 5-10`
- Adjust visualization: `contrast 3.5`

The terminal provides power-user control without cluttering the visual interface with buttons and menus.

### Inspection Panel

A collapsible side panel that appears when a neuron, head, or edge is selected. Shows detailed statistics, histograms, and connection maps for the selected component.

---

## Scope and Focus

Rikugan focuses exclusively on **transformer-based language models**. The architecture assumptions (residual stream, attention + MLP blocks, token-level processing) are specific to this family. Supporting other architectures (CNNs, RNNs, diffusion models) is out of scope for v1.

The tool is designed for **research and education**, not production monitoring. It prioritizes interpretability insight over performance benchmarking.

---

## What Doesn't Exist Yet

Existing tools cover parts of this space but none integrate them:

- **Netron**: static architecture visualization (graph of layers). No activations, no interactivity beyond navigation.
- **BertViz**: 2D attention pattern visualization. Single-purpose, no MLP analysis, no 3D.
- **TransformerLens / Baukit**: Python libraries for activation extraction and intervention. Powerful backends, no visual interface.
- **TensorBoard**: training metric dashboards. Not designed for single-inference interpretability.

Rikugan's contribution is the **integration**: architecture + activations + temporal playback + multi-scale navigation + interactive exploration, in a single coherent visual environment.

---

## Design Philosophy

The analogy is medical imaging. Rikugan offers the equivalent of:

- **MRI** at the macro level: see which regions of the model are most active for a given input, from multiple angles and cross-sections.
- **Electron microscopy** at the micro level: zoom into individual neurons and synaptic connections, observe signal propagation at the finest granularity.

The transition between these scales is continuous and lazy-loaded. The researcher never leaves the environment — they zoom, navigate, slice, and the data materializes around them.

---

*Document version 0.1 — Conceptual foundation and visual-mathematical framework.*
*Next: Technical architecture (stack, data structures, protocols).*
