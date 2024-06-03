# FMIPVerify.jl

_A package for evaluating the robustness of floating-point neural networks using Mixed Integer Programming (MIP). This project is extended from MIPVerify(https://github.com/vtjeng/MIPVerify.jl). See the [companion paper]() for more details._

## Getting Started

Due to the involvement of multiple packages, installation may be time-consuming.

### Prerequisites

To use our package, you will need:

1. The Julia programming language.
2. An optimizer supported by SageMath, which integrates the Rational number solver [PPL](https://doc.sagemath.org/html/en/reference/discrete_geometry/sage/geometry/polyhedron/backend_ppl.html) and the high-precision floating-point solver [Gurobi](https://www.gurobi.com/whats-new-gurobi-11-0/?utm_source=google&utm_medium=cpc&utm_campaign=2024+amer+googleads+awareness&campaignid=193283256&adgroupid=51266130904&creative=690777197021&keyword=gurobi&matchtype=b&_bn=g&gad_source=1&gclid=CjwKCAjwx-CyBhAqEiwAeOcTdUR58CeFO85-YVfhv0crTa-iGxQCszYv1Zj9_rPVBI2n19_FywaUdhoCzw8QAvD_BwE).
3. The Julia package for working with the optimizer.

### Installation on Ubuntu 20.04

Follow these steps to install FMIPVerify on Ubuntu 20.04:

1. **Download the FMIPVerify package**: Clone or download the package to your local environment.

2. **Install Julia 1.6.7**: Download and install Julia from [the official site](https://julialang.org/downloads/).

3. **Install Gurobi**: Download and install the most recent version of Gurobi from [Gurobi's official site](http://www.gurobi.com/downloads/gurobi-optimizer). A license is required to use Gurobi; free academic licenses are available [here](https://user.gurobi.com/download/licenses/free-academic).

4. **Install SageMath 10.2**: Follow the [official SageMath installation guide](https://doc.sagemath.org/html/en/installation/index.html) to install SageMath.

5. **Install FMIPVerify**:
    - Start the Julia REPL in the terminal:
      ```sh
      bash> julia
      ```
    - Enter the Pkg mode by pressing `]`:
      ```julia
      julia> ]
      ```
    - Use the `dev` command to install FMIPVerify from your local path:
      ```julia
      (pkg) dev path/to/FMIPVerify
      ```

6. **Install PyCall.jl**:
    - PyCall.jl is a Julia library for calling Python functions from Julia. We use this library to call SageMath, which is implemented based on Python.
    - You need to build PyCall.jl with the SageMath Python environment. Set the `PYTHON` environment variable to the path of the SageMath Python executable and build PyCall.jl:
      ```julia
      ENV["PYTHON"] = "... path to the SageMath Python executable ..."
      Pkg.build("PyCall")
      ```

### Project Structure

### Explanation of the Project Structure


1. **`Project.toml`**: Contains metadata about the project, including dependencies.
2. **`src/`**: Contains the source code of the package.
3. **`test/`**: Contains test scripts to ensure the package works as expected.
   - **`runtests.jl`**: The main script to run all tests.
4. **`experiments/`**: Contains all experiment code and results. You can check the experiment settings and experiments related to our work.
   - **`experiment2/robustness_check/float16_mnist_robust_check_verification_with_gurobi_itvbound77_epsilon0.ipynb`**: The name of this Notebook code tells us that it is used to verify whether FMIPVerify verifies the robustness of the 7 * 7 input image on the network when the disturbance is 0.
5. **`data/`**: Contains the MNIST dataset.
   - **`t10k-images-idx3-ubyte`**: The MNIST data file.
6. **`resized_images/`**: Contains the resized MNIST dataset.
   - **`resized_mnist_images77_test.bin`**: The resized 7*7 MNIST test data file.
7. **`deps/`**: Contains the training code and datasets for experiments.
   - **`train_floating16net_downsample_77.py`**: An example script for training the model.



### Overview of FMIPVerify Implementation
FMIPVerify translates your query on the robustness of a neural network for some input into an sound constraint MILP problem, which can then be solved by PPL for any Arbitrary precision floating-point FFNN, for low precision floating-point FFNN Gurobi solver can also maintain the soundness of validation results empirically compared to general real-number verification tools.


### Experiment setting and result

### Comparative Experimental Results: FMIPVerify and MIPVerify

| **Dataset**          | **Arch**         | **$\boldsymbol{\epsilon}$** | **n** | **FMIPVerify TN** | **FMIPVerify UK** | **MIPVerify TN+FN** | **MIPVerify TP** | **MIPVerify FP** |
|----------------------|------------------|----------------------------|-------|-------------------|-------------------|---------------------|------------------|------------------|
| **M<sup>H</sup>_{7×7}**  | **MLP<sup>H</sup>_a** | 0.0                        | 5000  | 4986              | 14                | 4999                | 0                | 1                |
| **M<sup>H</sup>_{28×28}** | **MLP<sup>H</sup>_b** | 0.0                        | 5000  | 4991              | 9                 | 4998                | 0                | 2                |
| **M<sup>H</sup>_{7×7}**  | **MLP<sup>H</sup>_a** | 0.05                       | 5000  | 374               | 4626              | 415                 | 2888             | 1697             |
| **M<sup>H</sup>_{28×28}** | **MLP<sup>H</sup>_b** | 0.05                       | 5000  | 1634              | 3366              | 1719                | 1736             | 1545             |
| **M<sup>H</sup>_{7×7}**  | **MLP<sup>H</sup>_a** | 0.1                        | 5000  | 3                 | 4997              | 3                   | 3925             | 1072             |
| **M<sup>H</sup>_{28×28}** | **MLP<sup>H</sup>_b** | 0.1                        | 5000  | 75                | 4925              | 81                  | 3083             | 1836             |


![Alt text](./experiments/experiments_summary/round_error_adv.png)








