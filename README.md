# Filtered Boris algorithms

Julia implementation of  "A ﬁltered Boris algorithm for charged-particle dynamics
in a strong magnetic ﬁeld" by Ernst Hairer, Christian Lubich, Bin Wang

Implemented by Carla López for the Physics Seminar of ETH Zurich

December 2024

## Repository Structure
```
BorisPusher
├── README.md
├── gitignore
├── Boris
|   ├── Manifest.toml
|   └── Project.toml
├── docs
│   ├── presentation.pdf
│   └── s00211-020-01105-3.pdf
├── figures
│   ├── log_error_log_e.pdf
│   └── log_error_log_he.pdf
├── scripts
│   ├── extras.jl
│   ├── plot_log_error_log_e.jl
│   └── plot_log_error_log_he.jl
└── src
    ├── integrators_one_step_map.jl
    ├── integrators_staggered.jl
    ├── matrix_funcs.jl
    ├── rodriguez.jl
    └── utils.jl
```
