# Demo experiments code

In this repository, we introduce our demo code. The main demo code can be viewed in notebook `demo/demo.ipynb`. Open the notebook and run the cells.
Below, in section [Additional experiments](#experiments) is an example of running.
To start any experiments, first go through all the installation steps from [Installation](#installation) section.

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
## Additional experiments<a name="experiments"></a>

For additional demo experiments we have implemented VAEs with different latent discrete distributions from **Just Relax It**. We borrowed it from the [pytorch repo](https://github.com/pytorch/examples/tree/main/vae). 
1. To run the additional demo code with VAEs, you should firstly train all the models and save their results. Run the following:
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
    ```
2. As you finished the training and testing of all the models, you can see the results of sampling and reconstruction methods in the directory `demo/results`.

Moreover, we conducted experiments with Laplace Bridge between LogisticNormal and Dirichlet distributions. We use two-side Laplace bridge to approximate:
- Dirichlet using logisticNormal
- LogisticNormal using Dirichlet
Thus, we find the best parameters to make the distributions almost the same on the simplex. These experiments can be found in the notebook `demo/laplace-bridge.ipynb`.

In addition, the Reinforce algorithm is applied in the [Acrobot environment](https://www.gymlibrary.dev/environments/classic_control/acrobot/). Detailed experiments can be viewed in the notebook `demo/reinforce.ipynb`. A script `demo/reinforce.py` can also be used for training.


