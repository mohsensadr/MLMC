![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# MLMC Implementation in Rust
This repository contains a **Rust implementation of the Multilevel Monte Carlo (MLMC) method** for solving stochastic differential equations (SDEs). It supports optimal and adaptive level selection and variance estimation via linear regression to determine MLMC parameters. For details of MLMC method, see [M. Giles paper](https://doi.org/10.1017/S096249291500001X).

![Demo](example/mlmc_vs_mc_cost.png)
![Demo](example/mlmc_Nl_eps.png)

## Features

- Simulates customizable SDEs.
- Computes MLMC estimators up to a specified tolerance `ε`.  
- Automatically estimates convergence rates via linear regression.  
- Outputs both **MLMC** and **standard Monte Carlo (MC)** costs.  

## Usage

Clone the repository:

```bash
git clone <repo_url>
cd <repo_dir>
```

Then, simply run

```bash
cargo run
```

You can modify the underlying SDE by changing the function
```
fn sde(l: usize, n: usize, sde_params: &SDEparams) -> [f64; 7] 
```

For plotting the output, use the jupyter notebook in ```example/``` directory as a example.
