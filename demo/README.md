# Demo experiments code

In this repository, we introduce our basic demo code. The main demo code can be viewed in notebook `demo/demo.ipynb`. Open the notebook and run the cells.
Below, we provide a detailed analysis of [additional demo experiments](#experiments).
To run any of the experiments, follow the [Installation](#installation) and [Usage](#usage) sections.

## Installation <a name="installation"></a>

To use this demo code, you need to have Python and the required packages installed on your computer.

Clone the repository:
```bash
git clone https://github.com/intsystems/discrete-variables-relaxation.git
```

Navigate to the repository directory:
```bash
cd discrete-variables-relaxation/demo
```

Create a conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate the conda environment:
```bash
conda activate relaxit-demo
```
## Additional experiments setup<a name="experiments"></a>

For additional demo experiments we have implemented a VAE. We borrowed it from the [pytorch repo](https://github.com/pytorch/examples/tree/main/vae). 

**Goal:** implement VAE with different latent discrete distributions

## Usage <a name="usage"></a>

1. To run the additional demo code, you should firstly train all the models and save their results. Run the following:
    ```bash
    # VAE with Gaussian Bernoulli latent space
    python vae_gaussian_bernoulli.py
    
    # VAE with Correlated Bernoulli latent space
    python vae_correlated_bernoulli.py
    
    # VAE with Hard Concrete latent space
    python vae_hard_concrete.py
    
    # VAE with Straight Through Bernoullii latent space
    python vae_straight_through_bernoulli.py

    # VAE with Invertible Gaussian latent space
    python vae_invertible_gaussian.py

    # VAE with Gumbel Softmax TopK latent space
    python vae_gumbel_softmax_topk.py

    # Reinforce training in the Acrobot environment
    python reinforce.py
    ```
2. As you finished the training and testing of all the models, you can see the results of sampling and reconstruction methods in the directory `demo/results`.
