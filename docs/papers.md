# Papers Using Prometheus

The following papers use Prometheus for neutrino telescope simulation. They are grouped by how they use Prometheus to give you an idea of what the package can be used for. For a complete and up-to-date list, see [INSPIRE HEP](https://inspirehep.net/literature?sort=mostrecent&size=25&page=1&q=refersto%3Arecid%3A2655303).

Want your paper featured on this list? Let us know in [discussions](https://github.com/Harvard-Neutrino/prometheus/discussions)!

## Detector Design and Optimization

These papers use Prometheus to simulate and evaluate new neutrino telescope geometries.

- [**POLARIS: A Sparse Radial Neutrino Telescope Design for the Pacific Ocean**](https://arxiv.org/abs/2604.12521) — Hymon et al. (2026) — Designs a sparse radial deep-water array optimized for horizontal muon tracks in the multi-TeV to PeV range, demonstrating competitive sensitivity with significantly fewer optical modules than general-purpose detectors.

- [**Comparison of Geometrical Layouts for Next-Generation Large-Volume Cherenkov Neutrino Telescopes**](https://arxiv.org/abs/2407.19010) — Zhu et al. (2024) — Systematically evaluates a range of optical module layouts at different geometrical volumes and assesses signal selection efficiency and reconstruction fidelity.

## Machine Learning and Event Reconstruction

These papers use Prometheus-generated simulations to develop and benchmark machine learning methods for neutrino telescopes.

- [**NuBench: An Open Benchmark for Deep Learning-Based Event Reconstruction in Neutrino Telescopes**](https://arxiv.org/abs/2511.13111) — Ørsøe et al. (2025) — Introduces an open benchmark comprising nearly 130 million simulated neutrino interactions across six detector geometries, for comparing machine learning reconstruction methods.

- [**Reducing Simulation Dependence in Neutrino Telescopes with Masked Point Modeling**](https://arxiv.org/abs/2510.01733) — Yu et al. (2025) — Presents the first self-supervised training pipeline for neutrino telescopes using point cloud transformers, shifting the majority of training to real data to reduce systematic uncertainties from simulation.

- [**GraphNeT 2.0: A Deep Learning Library for Neutrino Telescopes**](https://arxiv.org/abs/2501.03817) — Ørsøe et al. (2025) — Introduces a detector-agnostic deep learning library enabling inter-experimental collaboration on reconstruction methods across different neutrino telescope experiments.

- [**Enhancing Events in Neutrino Telescopes Through Deep Learning-Driven Superresolution**](https://arxiv.org/abs/2408.08474) — Yu et al. (2024) — Proposes a convolutional neural network technique to predict hits on virtual optical modules, demonstrating improved angular reconstruction in a simulated ice-based detector.

- [**Two Watts Is All You Need: Enabling In-Detector Real-Time Machine Learning for Neutrino Telescopes via Edge Computing**](https://arxiv.org/abs/2311.04983) — Jin et al. (2023) — Demonstrates the first deployment of machine learning methods on Google Edge TPUs for in-detector inference in water and ice neutrino telescopes.

## Companion Simulation Tools

These are simulation tools and software packages built to work alongside Prometheus.

- [**SIREN: An Open-Source Neutrino Injection Toolkit**](https://arxiv.org/abs/2406.01745) — Schneider et al. (2024) — Presents a tool for accurate neutrino interaction and detector geometry modeling that complements Prometheus for phenomenological studies of rare neutrino processes across a wide range of experimental designs.

## Quantum Computing Applications

These papers apply quantum computing techniques to neutrino telescope data, using Prometheus-generated simulations.

- [**Pathways in Neutrino Physics via Quantum-Encoded Data Analysis**](https://arxiv.org/abs/2402.19306) — Lazar et al. (2024) — Encodes neutrino telescope events into an IBM quantum processor using eight qubits and demonstrates a flavor classification task separating electron-neutrino from muon-neutrino events.

- **Quantum Contextual Memories** — Gatti Alvarez (2024, PhD thesis) — Introduces Quantum Contextual Memories as a framework for encoding and retrieving classical information within quantum systems, applied to neutrino telescope data.

## Reviews and Perspectives

These community review papers and white papers reference Prometheus as part of the broader neutrino telescope simulation landscape.

- [**From the Dawn of Neutrino Astronomy to a New View of the Extreme Universe**](https://arxiv.org/abs/2405.17623) — Argüelles et al. (2024) — Identifies seven major open questions in neutrino astrophysics and reviews experimental capabilities and proposed detector technologies.

- [**Machine Learning on Heterogeneous, Edge, and Quantum Hardware for Particle Physics (ML-HEQUPP)**](https://arxiv.org/abs/2602.22248) — Gonski et al. (2026) — A community white paper identifying research priorities in hardware-based machine learning for particle physics experiments.
