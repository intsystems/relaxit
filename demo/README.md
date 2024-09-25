# Demo experiments code

In this repository, we introduce our basic demo code. 
Below, we provide a detailed description of experiments setup.
To run the experiments, follow the [Installation](#installation) and [Usage](#usage) sections.

## Basic experiment setup

For basic demo experiments we have implemented a VAE. We borrowed it from the [pytorch repo](https://github.com/pytorch/examples/tree/main/vae). 

**Goal:** implement VAE with different latent distributions (continuous and discrete)

**Toolkit:** `torch.distributions` package

The experiment setup is as follows:
1. In `vae.py` we implement basic VAE with manual reparametrization and sampling using `torch.randn()`. Note, that latent variables distribution is gaussian.
2. Then we change the code using the `torch.distributions` package. Resulting script is provided in the `vae_gaussian.py`. Notice the changes in `forward` and `loss_function`. Here the gaussian latent distribution was used again.
3. The most interesting is that we can change the latent distribution to discrete one. In `vae_bernoulli.py` we used a Bernoulli distribution in latent space. We implement it using `RelaxedBernoulli` distribution.
4. Finally, we also implemented VAE with categorical latent space, which is available in `vae_categorical.py`. For this purpose, we used `RelaxedOneHotCategorical`.

## Installation <a name="installation"></a>

To use this demo code, you need to have Python and the required packages installed on your computer.

Clone the repository:
```bash
git clone https://github.com/intsystems/discrete-variables-relaxation.git
```

Navigate to the repository directory:
```bash
cd discrete-variables-relaxation/code
```

Create a conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate the conda environment:
```bash
conda activate relaxit-demo
```

## Usage <a name="usage"></a>

1. To run the basic demo code, you should firstly train all the models and save their results. Run the following:
    ```bash
    # basic VAE with Gaussian latent space
    python vae.py
    # basic VAE, but using `torch.distributions`
    python vae_gaussian.py
    # VAE with Bernoulli latent space
    python vae_bernoulli.py
    # VAE with Categorical latent space
    python vae_categorical.py
    ```
2. As you finished the training and testing of all the models, you can visualize the results using `visualization.ipynb`. Open the notebook and run the cells.