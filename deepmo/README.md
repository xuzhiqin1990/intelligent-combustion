# Project README

## Overview

This project provides a framework for optimizing chemical mechanisms using a combination of numerical simulations and machine learning techniques. It is designed to handle optimization tasks related to ignition delay time (IDT), perfectly stirred reactor (PSR), and laminar flame speed (LFS). The optimization process involves iterative sampling, data generation, and neural network training to refine the mechanism parameters.

## Directory Structure

The project requires a specific directory structure. Users should create their own working directory (e.g., `your_working_dir`), with the following subdirectories and files:

```
your_working_dir
    command
    settings
        setup.yaml           # Configuration file
        chem.yaml            # Mechanism file
        exp.csv              # Experimental data file (if needed)
```

- `your_working_dir`: Replace with your chosen working directory name.
- Some scripts have predefined maximum runtimes (e.g., for data generation). Adjust these as necessary based on your computational resources.

## Usage

1. Navigate to the `command` directory within your working directory:
   ```
   cd your_working_dir/command
   ```
2. Execute the bash script `transpose_server_FORpi.sh` to start the optimization process:
   ```
   bash transpose_server_FORpi.sh
   ```
   - Optional arguments:
     - `-s <i>`: Start from the i-th loop
     - `-e <i>`: End at the i-th loop
   - Example:
     ```
     bash transpose_server_FORpi.sh -s 4 -e 10
     ```
     This command starts the optimization from the 4th loop and ends at the 10th loop.

The script `transpose_server_FORpi.sh` automates data transfer between servers and sequentially executes modules in `main.py`, including:
- `sample_prepare`: Prepares samples for the optimization process.
- `nn_sample_gen` and `nn_sample_gen_MPI`: Generate neural network samples, utilizing MPI for parallel processing.
- `gather_data`: Collects and organizes the generated data.
- `inverse`: Performs the inversion step to optimize the mechanism parameters.

These modules leverage both CPU and GPU resources across two servers to efficiently handle computational tasks.

## Configuration

The optimization process is controlled by the `setup.yaml` configuration file located in the `settings` directory. Below is an example configuration:

```yaml
# Chemical Information
APART_base:
  reduced_mech: './settings/zhan2024_Nonegative.yaml'  # Path to the detailed mechanism
  IDT_mode: 1                                          # IDT mode
  idt_defined_species: 'OH'
  PSR_mode: False                                      # PSR mode; set to True to enable PSR training
  LFS_mode: True
  IDT_csv_path: './settings/IDT_ripe_data.csv'
  LFS_csv_path: './settings/laminarFS_ripe_data.csv'

# APART Arguments
APART_args:
  # Sampling Parameters
  SetAdjustableReactions_mode: 1
  GenASampleRange_mode: 'Scale_Sensitivity'
  sensitivity_scale_coeff: 3
  
  True_IDT_Cut_Time: 2
  # Parameters controlling the overall sampling
  father_sample_size: 1
  sample_size: 50000

  # Parameters controlling the Net-Sample selection rules
  l_alpha: 0.15    # Enabled when GenASampleRange_mode is set to None; left sampling boundary
  r_alpha: 0.15    # Enabled when GenASampleRange_mode is set to None; right sampling boundary
  idt_threshold: [10, 8, 6, 5, 4, 3, 2.5, 2, 2, 1.8, 1.6]
  lfs_threshold: [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, ]
  T_threshold_ratio: 0.95
  IDT_weight: 1
  LFS_weight: 0.8
  passing_rate_upper_limit: 0.4
  threshold_expand_factor: 2
  PSRex_cutdown: False

  # Training neural networks during iterations
  shrink_strategy: True
  shrink_delta: 0.02
  shrink_ratio: 0.0
  extract_stategy: True
  extract_ratio: 0.05
  concat_pre: False                          # Whether to use previous apart_data for training
  reuse_net: False                           # Whether to continue training with the previous network
  optimizer: 'Adam'                          # Optimizer for training
  learning_rate: 2.0e-5                      # Learning rate for training
  lr_decay_step: 20                          # Number of epochs before learning rate decay
  lr_decay_rate: 0.95                        # Learning rate decay factor
  batch_size: 1000                           # Batch size
  hidden_units: [3000, 2000, 2000]           # Width of hidden layers; IDT and PSR networks are trained separately
  epoch: 800
```

Users should adjust these parameters according to their specific optimization needs and computational resources.

## Data Storage and Results

After executing the script, an `inverse_skip` folder will be created in the working directory. This folder contains subfolders for each optimization loop (e.g., `circ=1`, `circ=2`, etc.), each storing:

- The top 15 optimal parameter combinations.
- The optimized mechanism file `optim_chem.yaml` (typically, the one in `circ=i/0` is considered the best for that loop).

Additionally, visualization files such as `compare_nn_IDT.png` and `compare_nn_LFS.png` are generated to illustrate the optimization effects:

- **Red crosses**: Errors of the mechanism before optimization.
- **Blue crosses**: Errors after optimization.
- **Blue circles**: Prediction errors of the deep neural network (DNN) after optimization.

The proximity of the blue crosses to the center line indicates the effectiveness of the optimization, while the alignment of blue crosses with blue circles reflects the accuracy of the network training.
