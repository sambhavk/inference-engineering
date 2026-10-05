# Inference Engineering

A collection of notebooks tinkering around inferencing concepts. Each notebook is self-contained.

## Requirements

- **NVIDIA GPU with tensor cores only (compute capability ≥ 7.0).** All experiments use CUDA and PyTorch; AMD, Apple Metal/MPS and other accelerators are not supported.
- Notebooks are GPU-agnostic: set your GPU's spec-sheet peak FLOP/s and bandwidth in each notebook's hardware section. Tested on Google Colab (T4).
- Without a CUDA device, notebooks still run on CPU for correctness checks, but performance numbers are not meaningful.

## Index

| # | Topic | Notebook |
|---|-------|----------|
| 01 | Op fusion and arithmetic intensity | [01_op_fusion_arithmetic_intensity.ipynb](notebooks/01_op_fusion_arithmetic_intensity.ipynb) |
| 02 | Vertical fusion of elementwise ops (`torch.compile`) | [02_vertical_fusion_elementwise.ipynb](notebooks/02_vertical_fusion_elementwise.ipynb) |
| 03 | GPU benchmarking: peak FLOP/s, power cap, bandwidth, roofline sweep | [03_gpu_benchmarking.ipynb](notebooks/03_gpu_benchmarking.ipynb) |
