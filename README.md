# A Toy Model of Superposition — a statistical-physics view

**How can a neural network remember more things than it has room for? It cheats — the same way atoms, magnets, and packed spheres do.**

This repository builds the smallest neural network that exhibits *superposition*, reproduces the key results from scratch, and then analyses them with the tools of statistical physics: order parameters, phase transitions, and energy minimisation. Everything runs on a laptop CPU in a few minutes.

---

## 1. The one-paragraph version (no background assumed)

A neural network stores information inside a fixed number of internal "slots" (think of them as the axes of a coordinate system). Common sense says it can store at most one piece of information per slot. **Superposition** is the surprising trick where a network stores *many more* pieces of information than it has slots, by laying them down as *overlapping* directions that share the slots. This works only because the information is **sparse** — most pieces are "silent" at any given moment, so the overlaps rarely cause confusion. Below we watch a network discover this trick on its own, and we show that the shapes it settles into are exactly the shapes a physicist would predict from an energy-minimization argument.

### A few terms, defined once

- **Feature**: one independent piece of information the network might want to represent (e.g. "the image contains a wheel"). We use *n* features.
- **Dimension / slot**: one axis of the network's internal storage space. We use *m* of them, with **m < n** — fewer slots than features. That scarcity is the whole point.
- **Sparsity**: the fraction of the time a feature is *off* (exactly zero). Sparsity 0 = every feature is always on ("dense" world); sparsity near 1 = features almost never appear together ("sparse" world). This is our master control knob.
- **Superposition**: storing more features than dimensions by using overlapping, non-perpendicular directions.

---

## 2. Why this is really a physics problem

If you must place *n* feature directions in only *m* dimensions, they cannot all be mutually perpendicular — there isn't enough room. So they interfere. Each pair of features that points in similar directions pays a penalty (they get confused for one another). The network wants to arrange the directions to make the *total* penalty as small as possible.

That is a **frustration problem**, and physicists have studied its cousins for over a century:

- It is the **Thomson problem**: place electrons on a sphere so that their mutual repulsion is minimized. The solutions are beautiful regular shapes.
- It is **sphere packing** and **spin frustration**: many parts, each wanting to avoid the others, settling into an ordered compromise.

And because we have a control knob (sparsity), we can ask the question at the heart of statistical mechanics: **does the system change phase as we turn the knob?** It does — sharply — and the ordered phases are regular polygons.

This framing is the contribution of this repo on top of the original result: superposition is a *packing transition*, and the language of order parameters and phase diagrams describes it cleanly.

---

## 3. The model

The network is a deliberately minimal **autoencoder** (a network trained to copy its input to its output through a narrow bottleneck):

```
h  = W x              compress n features into m < n dimensions
x' = ReLU(Wᵀ h + b)   reconstruct the n features
```

There is a single weight matrix `W` of shape `(m, n)`. **Column *i* of `W` is the direction the network uses to store feature *i*** — call it that feature's representation vector. `ReLU` (Rectified Linear Unit) is the function `max(0, ·)`; it zeroes out negatives and is what lets the network suppress the small interference from overlapping features.

We train on synthetic data: each example is a vector of *n* features, each independently off with probability = sparsity, and drawn from `[0,1)` when on. The model never sees a fixed dataset — it learns the *statistics* of the sparse world. The loss is the (importance-weighted) squared reconstruction error.

The full engine is ~200 lines in [`src/superposition/`](src/superposition).

---

## 4. Results

### Experiment 1 — superposition emerges as the world gets sparser

`python experiments/01_superposition_emergence.py`

We use the tiniest interesting model: **n = 5 features, m = 2 dimensions**, so we can *draw* the storage space on a flat page. Each arrow is one feature's representation vector; the bar below counts how many dimensions each feature effectively occupies.

![Emergence of superposition](figures/01_superposition_emergence.png)

- **Dense world (sparsity 0):** the network can only afford **2** of the 5 features and stores them on **perpendicular axes** — no overlap, no superposition. The other three features are dropped.
- **As sparsity rises:** the network squeezes in more features. At intermediate sparsity it stores **4** features as two perpendicular pairs (a "plus" sign).
- **Sparse world (sparsity ≈ 0.97):** it stores **all 5** features as a perfect **regular pentagon**. The directions overlap, but since features rarely co-occur, the overlap rarely costs anything.

This is the canonical "Toy Models of Superposition" result, reproduced from scratch.

### Experiment 2 — a phase diagram

`python experiments/02_phase_diagram.py`

Now a bigger model (**n = 40 features, m = 10 dimensions**). We pick an **order parameter** — a single number summarising the macroscopic state — and sweep the sparsity knob. Our order parameter is **features-per-dimension**: how many features the network actually stores, divided by the number of dimensions. A value of 1 means "no superposition" (each stored feature owns a dimension); a value above 1 means superposition.

![Phase diagram](figures/02_phase_diagram.png)

The network sits in a **no-superposition phase** (features-per-dimension ≈ 1) in the dense world, then transitions into a **superposition phase** as the world gets sparse, packing several features into every dimension. This is structurally identical to a magnet ordering as temperature drops: an order parameter responding to a control parameter, with a recognisable transition region.

### Experiment 3 — the geometry is quantised, and it solves a packing problem

`python experiments/03_feature_geometry.py`

Back to the n = 5, m = 2 model, sweeping sparsity finely.

![Feature geometry and packing](figures/03_feature_geometry.png)

**(A) Quantised geometry.** The dimensionality of a feature does not drift smoothly — it **locks onto a discrete ladder** of values, each one corresponding to a specific regular shape:

| Dimensionality `Dᵢ` | Shape the features form |
|---|---|
| `1`   | a feature alone on its own axis |
| `1/2` | two features back-to-back (a pair) |
| `2/5` | five features at 72° (a pentagon) |

The flat plateaus separated by jumps are the signature of distinct **geometric phases** — exactly like discrete energy levels or magnetisation plateaus in a physical system. (The `2/3` triangle line is drawn for reference; this particular model jumps past it.)

**(B) A solved packing problem.** At high sparsity the five features form a regular pentagon. We define a **frustration energy** — the total squared overlap between feature directions, the network's analogue of electrostatic repulsion — and compare the trained network to the ideal pentagon:

> **learned energy = 3.750, ideal regular-pentagon energy = 3.750** — they match to numerical precision.

The network, simply by minimizing reconstruction error, has independently found the minimum-energy packing — the same answer the Thomson problem gives for 5 points on a circle.

---

## 5. Why this matters (and why I built it)

Modern language models almost certainly use superposition heavily: they represent vastly more concepts than they have neurons. That is a central obstacle for **interpretability** — the effort to understand what a network has actually learned — because a single neuron ends up participating in many unrelated concepts. Understanding superposition in a setting we can fully solve is a prerequisite for understanding it where it matters.

My background is in soft-matter and statistical-mechanics physics, where frustration, packing, and phase transitions are everyday tools. This project is a demonstration that those tools transfer directly: superposition is a packing transition, and it yields to the same analysis I would apply to colloids on a sphere or spins on a lattice.

---

## 6. Run it yourself

```bash
git clone <this-repo>
cd Superposition
pip install -e .            # installs torch, numpy, matplotlib
python experiments/01_superposition_emergence.py
python experiments/02_phase_diagram.py
python experiments/03_feature_geometry.py
pytest                      # fast sanity tests
```

Figures are written to [`figures/`](figures). Everything runs on CPU in a few minutes.

## 7. Repository layout

```
src/superposition/
  data.py      sparse synthetic features (the "world")
  model.py     the toy autoencoder
  train.py     training loop + config
  metrics.py   order parameters: dimensionality, frustration energy
  viz.py       plotting helpers
experiments/   three scripts, each producing one figure above
tests/         fast checks of the engine and metrics
```

## 8. References

- Elhage et al., **"Toy Models of Superposition"**, Anthropic / Transformer Circuits Thread, 2022. <https://transformer-circuits.pub/2022/toy_model/index.html>
- J. J. Thomson (1904) and the long line of work on minimum-energy point configurations on the sphere (the Thomson / Tammes problems).

---

*Built as a research demonstration. The physics framing — order parameters, the phase diagram, and the explicit Thomson-problem connection — is my own addition on top of the reproduced toy model.*
