# Project: 2-dimensional Monte-Carlo simulation in canonical ensemble

This project focuses on the $\text{He}^3$ - $\text{He}^4$ mixture simulation. It is optimized for high performance using **Numba** (CPU) and **CuPy** (GPU). 

The coding process is assisted by **GEMINI Cli**, which provides code suggestions and algorithmic optimizations.

## Requirements
- Python 3.10+
- NumPy, Matplotlib, SciPy, tqdm
- **Numba**: For CPU loop acceleration.
- **CuPy**: For GPU acceleration (requires NVIDIA GPU and CUDA, not mandatory for CPU simulations).

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

### 0. Construction of Phase Diagram
Go to ```src/Problem2.ipynb``` to run the code that generates different figures needed for Ising model analysis.

### 1. CPU Simulation (Optimized with Numba)
To run a single simulation with parameter-based output naming:
```bash
python src/main.py --N 40 --x 0.5 --j 1.0 --niters 1000 --nndiag --w_diag 1.0
```
- **N**: Edge Size (Default: 40).
- **niters**: Total steps (Default: 1000).
- **nndiag**: Enable diagonal neighbors.
- **w_diag**: Interaction weight for diagonal neighbors (Default: 0.5).
- **Output Path**: `output/sim_N{N}_x{x}_j{j}_iters{niters}_diag{T/F}_w{w_diag}/`

### 2. GPU Simulation (Experimental, Not Ideal Now)
> **Note**: The GPU version is an experimental prototype and may not cover all edge cases or physical behaviors identically to the CPU version.
```bash
python src/run_gpu.py --N 100 --niters 2000 --j 1.5 --nndiag --w_diag 1.0
```

### 3. Concentration Analysis (Parallel Processing)
Scan magnetization $\left< m \right>$ vs concentration $x$ using multiple CPU cores:
```bash
python src/analyze_x.py --N 40 --j 1.0 --niters 10000 --x_steps 11 --nprocs 8 --eq_start 0.5 --nndiag --w_diag 1.0
```
- **nprocs**: Number of parallel processes to use.
- **eq_start**: Fraction of steps to skip before averaging (Default: 0.5, i.e., average the second half).

## Features
- **Numba JIT Acceleration**: Core loops are compiled to machine code, achieving ~6x speedup over standard Python.
- **Multi-core Parallelism**: Analyze multiple concentrations simultaneously.
- **Dynamic Naming**: Results are automatically organized into folders named after their simulation parameters.

## Visualization Style
- **Square Plots**: All E, m, and configuration plots use a 1:1 aspect ratio.
- **Grid-Filling**: Configurations are plotted as solid cells (Red: +1, Green: -1, White: 0).
