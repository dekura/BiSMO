<p align="center">
  <h1>BiSMO: Bilevel Source Mask Optimization</h1>
</p>

## Overview

BiSMO is a PyTorch-based framework for bilevel optimization in lithography applications, specifically for Source Mask Optimization (SMO). It provides a modular and efficient approach to solve the complex optimization problems in lithography by leveraging bilevel optimization techniques.

The framework is built on top of [Betty](https://github.com/leopard-ai/betty), an automatic differentiation library for generalized meta-learning and multilevel optimization.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BiSMO.git
cd BiSMO

# Create and activate a conda environment (recommended)
conda create -n smo python=3.8
conda activate smo

# Install dependencies
pip install -r requirements/requirements.txt
```

## Quick Start

BiSMO supports multiple bilevel optimization approaches:

- **DARTS**: Differentiable Architecture Search method
- **NMN**: Neumann Series method
- **CG**: Conjugate Gradient method
- **RL**: Reinforcement Learning approach

To run an experiment with one of these approaches:

```bash
# Using DARTS approach
./darts_d2.sh

# Using Neumann Series approach
./nmn_d1.sh

# Using Conjugate Gradient approach
./cg_d0.sh

# Using Reinforcement Learning approach
./rl_d1.sh
```

## Structure

The project is organized as follows:

- `src/`: Main source code
  - `bilevel.py`: Entry point for bilevel optimization
  - `betty/`: Betty library integration
  - `models/`: Neural network models
    - `mo_module.py`: Mask Optimization module
    - `so_module.py`: Source Optimization module
  - `problems/`: Optimization problems
    - `mo.py`: Mask Optimization problem
    - `so.py`: Source Optimization problem
  - `engine/`: Optimization engines
  - `data/`: Data handling utilities

- `configs/`: Configuration files
  - `problems/`: Configuration for different optimization approaches
  - `module/`: Model configurations
  - `engine/`: Engine configurations

- Shell scripts (e.g., `darts_d2.sh`, `nmn_d1.sh`, etc.): For running experiments with different configurations

## Configuration

BiSMO uses [Hydra](https://hydra.cc/) for configuration management. Main configuration options include:

- `problem_type`: Optimization approach (`darts`, `nmn`, `cg`, `rl`)
- `unroll_steps`: Number of unrolling steps in optimization
- `device_id`: GPU device to use
- Weights for different loss components
- Learning rates and other hyperparameters

## Running Experiments

The provided shell scripts run optimization on different image masks with various approaches:

```bash
# Run optimization on 10 masks using DARTS approach on GPU 2
./darts_d2.sh

# Run with different unrolling steps (e.g., 3 steps) using Neumann Series
./nmn_d1_unroll3_iter.sh

# Run with different Conjugate Gradient iterations
./cg_d0_unroll3_iter.sh
```

## Customization

You can modify the configuration files in `configs/` to customize:

- Optimization approaches and hyperparameters
- Model architectures
- Input data sources and masks
- Training and validation settings

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{
  author = {Chen, Guojin and others},
  title = {BiSMO: Bilevel Source Mask Optimization},
  booktitle = {},
  year = {2023}
}
```

## License

[License information]

## Contact

Guojin Chen (cgjcuhk@gmail.com)
Homepage: https://gjchen.me
