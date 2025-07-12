# Paper Implementations (PyTorch)

This repository is a collection of clean, production-ready PyTorch implementations of influential deep learning papers. Each paper is organized in its own directory at the root of the repository.

## Structure

- Each folder at the root corresponds to a single paper (e.g., `resnet/`, `vit/`, etc.)
- Each paper directory contains its own code, README, requirements, and scripts

Example:
```
paper-implementations/
├── resnet/
│   ├── README.md
│   ├── requirements.txt
│   ├── models/
│   └── ...
├── vit/
│   ├── README.md
│   └── ...
└── README.md  # (this file)
```

## How to Add a New Paper
1. Create a new directory at the root named after the paper or model (e.g., `densenet/`)
2. Add your implementation, a README, and any requirements or scripts
3. Follow the structure of existing folders for consistency

## Current Papers
- `resnet/`: Deep Residual Learning for Image Recognition (He et al., 2015)

## Goals
- Provide reference implementations for classic and modern deep learning papers
- Make it easy to compare, extend, and experiment with different architectures
- Encourage reproducibility and clarity in research code

## License
MIT License 