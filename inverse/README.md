# Inverse Dynamics Training

This directory contains code for training inverse dynamics models that learn to predict actions given state transitions (s, s') → a.

## Files

- `learned_inverse_dynamics.py`: Contains the FlowInverseDynamics model implementation
- `dataset_creator_id.py`: Creates inverse dynamics datasets from sequence datasets
- `train_id.py`: Trains inverse dynamics models
- `test_train_id.py`: Simple test script (requires torch)

## Usage

### 1. Create Inverse Dynamics Dataset

First, create a dataset containing (s, a, s') triplets from a sequence dataset:

```bash
cd /home/wph52/weird/dynamics/inverse
python dataset_creator_id.py --config ../configs/dataset/pendulum_integrable_dynamics_shift.yaml --max_samples 1000
```

This will create a dataset at the path specified in the config, replacing `/sequence/` with `/inverse_dynamics/`.

### 2. Train Inverse Dynamics Model

Train the model using the created dataset:

```bash
python train_id.py \
    --dataset_config ../configs/dataset/pendulum_integrable_dynamics_shift.yaml \
    --model_config ../configs/inverse_dynamics/pendulum_flow.yaml \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --batch_size 64 \
    --device cuda
```

### 3. Model Configuration

The model configs are in `../configs/inverse_dynamics/`. Example config:

```yaml
name: "pendulum_flow_inverse_dynamics"
_target_: "dynamics.inverse.learned_inverse_dynamics.FlowInverseDynamics"
diffusion_step_embed_dim: 16
down_dims: [16, 32, 64]
num_inference_steps: 100
device: "cuda"
```

## Model Architecture

The `FlowInverseDynamics` model uses a conditional flow model to learn p(a | s, s'):

- **Input**: Current state (s) and next state (s')
- **Output**: Action (a) that would cause the transition s → s'
- **Architecture**: Conditional flow model that flows from N(0,1) to action space, conditioned on state transition

## Key Differences from Action Translation

1. **Data**: Uses (s, a, s') triplets instead of (s, a_original, a_shifted)
2. **Model**: Predicts actions from state transitions instead of translating actions
3. **Purpose**: Learn inverse dynamics for a single environment instead of translating between environments
