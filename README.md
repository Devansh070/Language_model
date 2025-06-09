# MiniGPT Language Model

A lightweight implementation of a GPT-style language model using TensorFlow, featuring:
- Transformer architecture with rotary positional embeddings
- Mixture of Experts (MoE) for efficient computation
- Configurable model size and training parameters
- Support for custom datasets

## Model Architecture
- Embedding dimension: 256
- Number of attention heads: 4
- Number of transformer layers: 8
- Feed-forward dimension: 768
- Number of experts: 4

## Setup
1. Install dependencies:
```bash
pip install tensorflow numpy
```

2. Train the model:
```bash
python train_minigpt.py
```

## Configuration
Model and training parameters can be configured in `training_config.json`:
- Batch size
- Learning rate
- Number of epochs
- Sequence length
- And more...

## Files
- `minigpt_transformer.py`: Core model implementation
- `train_minigpt.py`: Training script
- `training_config.json`: Training configuration 