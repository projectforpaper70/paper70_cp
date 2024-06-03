# Documentation for Project Artifacts

## Project Datasets

### Dataset 1: $M_{7 \times 7}^H$
- **Description**: This dataset includes compressed 7x7 MNIST inputs, covering both training and test sets.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/resized_images`

### Dataset 2: $M_{28 \times 28}^H$
- **Description**: Consisting of the standard 28x28 MNIST dataset, including training and test sets.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/datasets/mnist`

### Dataset 3: $FM_{28 \times 28}^H$ (newly added)
- **Description**: Fashion MNIST dataset, including training and test sets.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/datasets/fashionmnist`

## Models

### Model 1: $MLP_a^H$
- **Description**: A feedforward neural network [49, 10, 10, 10] trained on $M_{7 \times 7}^H$.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/weights/mnist/resized_mnist77input_mnist_dnn_fp16.mat`
- **Training Code**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/nnet_model/mnist/train_floating16net_downsample_77.py`

### Model 2: $MLP_b^H$
- **Description**: A feedforward neural network [784, 24, 24, 10] trained on $M_{28 \times 28}^H$.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/weights/mnist/mnist_dnn_fp16.mat`
- **Training Code**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/nnet_model/mnist/train_floating16net.py`

### Model 3: $MLP_c^H$ (newly added)
- **Description**: A feedforward neural network [784, 24, 24, 10] trained on $FM_{28 \times 28}^H$.
- **Location**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/weights/mnist/fashion_mnist_model.mat`
- **Training Code**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/deps/weights/mnist/fashmnist/fashion_mnist_fp16_train.py`

## Source Code Implementation
All source code implementations related to FMPVerify are located in the `Src` folder:
`/home/aritifact/aritifact_for_cp24/FloatMIPVerify/src`

## Implementation Details
All code implementations related to the paper are located in:
`/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments`.

To replicate our project implementation, follow the steps outlined below. All parameters and configuration files are set correctly. However, if you wish to use Gurobi as the solver or compare with MIPVerify implementation (also based on Gurobi), you need to apply for a Gurobi academic account [here](https://www.gurobi.com/academia/academic-program-and-licenses/) and reactivate Gurobi.







# Experiment 1: Analyzing the Soundness of Interval Abstract

## 1. Experimental Code
- **Bound precision for MLP_a**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment1/float16_interval_effectness.ipynb`
- **Bound precision for MLP_b**: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment1/float32_interval_effectness.ipynb`

## 2. Code Explanation
In these Jupyter notebooks, we demonstrate the implementation of pure interval propagation. The interval arithmetic methods used are consistent with Equation (16) in our paper.

## 3. Execution
To run the notebooks, use the following VSCode command:

### Bound precision for MLP_a:
```sh
code /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment1/float16_interval_effectness.ipynb
```

Click "Run" (our original run results have not been cleared, so you can observe that they are consistent with the implementation in the paper).


### Bound precision for MLP_b:
```sh
code /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment1/float32_interval_effectness.ipynb
```

# Experiment 2: FMIPVerify Vs MIPVerify Robustness Verification Experiment

All files related to Experiment 2 are located in: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2`. In this folder, we offer two modes of execution: command-line and VS Code notebook. Each experiment has a dedicated file in the VS Code notebook format for running our experiments. The file names are self-explanatory, requiring minimal explanation. We primarily focus on elucidating the command-line execution method and parameter settings.

## 1. FMIPVerify Robustness Verification Experiment

### (1) Experimental Code
Path: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check/FMIP_robustness_verification.jl`

### (2) Code Explanation
In this code, we accept parameters to run the experiment, including the dataset, model, perturbation radius, etc. The code returns verification results, and FMIPVerify prints the training results for each sample during the verification process, including `verify_true` and `verify_unknown`. The results for both verification outcomes are recorded in a file.

### (3) Parameter Explanation
1. `data_type` (Int) - e.g., 1 or 2
2. `data_name` (String) - should be "MNIST", "mnist49", or "FASHIONMNIST"
3. `network_name` (String) - should be "F16MNISTinput_77", "F16MNIST24", or "F16Fashion24"
4. `itv_network_name` (String) - should be "F16MNISTinput_77itv", "F16MNIST24itv", or "F16Fashion784itv"
5. `min_data_num` (Int) - [min_data_num, max_data_num] represents the range of the verification dataset, e.g., 0
6. `max_data_num` (Int) - e.g., 10000
7. `epsilon` - perturbation radius, e.g., 0, 0.05, 0.1
8. `solver` (optional, String) - "ppl" or "gurobi"

### (4) Experimental Setup
We conducted 7 experiments in FMIPVerify, each targeting robustness verification under different datasets and perturbations.

1. **Experiment 1**
   - **Settings:**
     - `data_name`: mnist49
     - `network_name`: F16MNISTinput_77
     - `numrange`: 1-5622
     - `epsilon`: 0.0
     - `solver`: ppl
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 F16MNISTinput_77itv 1 5622 0.0 PPL
     ```

2. **Experiment 2**
   - **Settings:**
     - `data_name`: mnist49
     - `network_name`: F16MNISTinput_77
     - `numrange`: 1-5622
     - `epsilon`: 0.05
     - `solver`: ppl
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 F16MNISTinput_77itv 1 5622 0.05 PPL
     ```

3. **Experiment 3**
   - **Settings:**
     - `data_name`: mnist49
     - `network_name`: F16MNISTinput_77
     - `numrange`: 1-5622
     - `epsilon`: 0.1
     - `solver`: ppl
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 F16MNISTinput_77itv 1 5622 0.1 PPL
     ```

4. **Experiment 4**
   - **Settings:**
     - `data_name`: MNIST
     - `network_name`: F16MNIST24
     - `numrange`: 1-5513
     - `epsilon`: 0.0
     - `solver`: gurobi
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 2 MNIST F16MNIST24 F16MNIST24itv 1 5513 0.0 gurobi
     ```

5. **Experiment 5**
   - **Settings:**
     - `data_name`: MNIST
     - `network_name`: F16MNIST24
     - `numrange`: 1-5513
     - `epsilon`: 0.05
     - `solver`: gurobi
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 2 MNIST F16MNIST24 F16MNIST24itv 1 5513 0.05 gurobi
     ```

6. **Experiment 6**
   - **Settings:**
     - `data_name`: MNIST
     - `network_name`: F16MNIST24
     - `numrange`: 1-5513
     - `epsilon`: 0.1
     - `solver`: gurobi
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 2 MNIST F16MNIST24 F16MNIST24itv 1 5513 0.1 gurobi
     ```

7. **Experiment 7** (not yet verified for Fashion MNIST)
   - **Settings:**
     - `data_name`: FASHIONMNIST 
     - `network_name`: F16Fashion24
     - `numrange`: 1-6000
     - `epsilon`: 0.1
     - `solver`: gurobi
   - **Command:**
     ```bash
     cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
     julia FMIP_robustness_verification.jl 2 FASHIONMNIST F16Fashion24 F16Fashion784itv 1 6000 0
     ```

## MIPVerify Robustness Verification

### (1) Experimental Code
Path: `/home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check/MIP_robustness_verification.jl`

### (2) Code Explanation
In this code, we accept parameters to run the experiment, including the dataset, model, perturbation radius, etc. The code returns verification results, and MIPVerify prints the training results for each sample during the verification process, including `verify_true`, `verify_false`, and `verify-unknown`. The results for both verification outcomes are recorded in a file.

### (3) Parameter Explanation
1. `data_type` (Int) - e.g., 1 or 2
2. `data_name` (String) - should be "MNIST", "mnist49", or "FASHIONMNIST"
3. `network_name` (String) - should be "F16MNISTinput_77", "F16MNIST24", or "F16Fashion24"
4. `min_data_num` (Int) - [min_data_num, max_data_num] represents the range of the verification dataset, e.g., 0
5. `max_data_num` (Int) - e.g., 10000
6. `epsilon` - perturbation radius, e.g., 0, 0.05, 0.1

### (4) Experimental Setup
We conducted 6 control experiments in MIPVerify, each targeting robustness verification under different datasets and perturbations.

#### 1. Experiment 1
**Parameter Settings:**
- `data_name`: mnist49
- `network_name`: F16MNISTinput_77
- `numrange`: 1-5622
- `epsilon`: 0.0

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 1 5622 0.0
```

#### 2. Experiment 2

**Parameter Settings:**
- `data_name`: mnist49
- `network_name`: F16MNISTinput_77
- `numrange`: 1-5622
- `epsilon`: 0.05

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 1 5622 0.05
```

### 3. Experiment 3

**Parameter Settings:**
- `data_name`: mnist49
- `network_name`: F16MNISTinput_77
- `numrange`: 1-5622
- `epsilon`: 0.1

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 1 mnist49 F16MNISTinput_77 1 5622 0.1
```

### 4. Experiment 4

**Parameter Settings:**
- `data_name`: MNIST
- `network_name`: F16MNIST24
- `numrange`: 1-5513
- `epsilon`: 0.0

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 2 MNIST F16MNIST24 1 5513 0.0

```

### 5. Experiment 5

**Parameter Settings:**
- `data_name`: MNIST
- `network_name`: F16MNIST24
- `numrange`: 1-5513
- `epsilon`: 0.05

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 2 MNIST F16MNIST24 1 5513 0.05
```

### 6. Experiment 6

**Parameter Settings:**
- `data_name`: MNIST
- `network_name`: F16MNIST24
- `numrange`: 1-5513
- `epsilon`: 0.1
- `solver`: gurobi

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 2 MNIST F16MNIST24 1 5513 0.1
```

### 7. Experiment 7 (Not yet verified, MIP performance on Fashion MNIST)

**Parameter Settings:**
- `data_name`: FASHIONMNIST
- `network_name`: F16Fashion24
- `numrange`: 1-6000
- `epsilon`: 0.1
- `solver`: gurobi

**Execution Command:**
```sh
cd /home/aritifact/aritifact_for_cp24/FloatMIPVerify/experiments/experiment2/mnist_robustness_check
julia MIP_robustness_verification.jl 2 FASHIONMNIST F16Fashion24 1 6000 0
```

## Results 
This is the result obtained from running under the configuration of our own paper Sect 5.1 Benchmark environment. If interested, you can verify it yourself.


| dataset                 | model         | $\epsilon$ | $n$   | FMIPVerify $\mathrm{TN}$ | FMIPVerify UK | MIPVerify $\mathrm{TN}+\mathrm{FN}$ | MIPVerify $\mathrm{TP}$ | MIPVerify FP |
|-------------------------|---------------|------------|-------|--------------------------|---------------|--------------------------------------|--------------------------|--------------|
| $\operatorname{MNIST}_{7 \times 7}^{H}$ | $\operatorname{MLP}_{a}^{H}$ | 0          | 5000  | 4975                     | 25            | 5000                                 | 0                        | 0            |
| $\operatorname{MNIST}_{28 \times 28}^{H}$ | $\mathrm{MLP}_{b}^{H}$ | 0          | 5000  | 4987                     | 13            | 4998                                 | 0                        | 2            |
| $\operatorname{MNIST}_{7 \times 7}^{H}$ | $\operatorname{MLP}_{a}^{H}$ | 0.05       | 5000  | 367                      | 4633          | 415                                  | 2880                     | 1705         |
| $\operatorname{MNIST}_{28 \times 28}^{H}$ | $\mathrm{MLP}_{b}^{H}$ | 0.05       | 5000  | 1516                     | 3484          | 1719                                 | 1736                     | 1545         |
| $\operatorname{MNIST}_{7 \times 7}^{H}$ | $\operatorname{MLP}_{a}^{H}$ | 0.1        | 5000  | 2                        | 4998          | 3                                    | 3926                     | 1071         |
| $\operatorname{MNIST}_{28 \times 28}^{H}$ | $\mathrm{MLP}_{b}^{H}$ | 0.1        | 5000  | 73                       | 4927          | 81                                   | 3083                     | 1836         |
| $\mathrm{FSMNIST}_{28 \times 28}^{H}$ | $\mathrm{MLP}_{c}^{H}$ | 0          | 4267  | 4223                     | 44            |                                      |                          |              |




