---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# SMC Pretuning

```{code-cell} ipython3
:tags: [remove-output]

import jax
from jax import numpy as jnp
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools
from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{raw-cell}
This notebook is a continuation of `Use Tempered SMC to Improve Exploration of MCMC Methods` and `Tuning inner kernel parameters of SMC`. In those notebooks, we sampled from a multimodal distribution with SMC, using HMC as inner kernel for particle mutation. Moreover, we started with all particles sharing fixed parameters, then we tuned the value of a parameter based on some estimator on top of the particle's population. 

A further refinement first developed in https://arxiv.org/abs/1808.07730, is to compute a probability distribution of a parameter, and assign a different sample to the chain associated on each particle. The way the algorithm works is by:

0. sampling an initial set of samples for the parameters of the chains. One sample per particle.
1. taking some MCMC steps starting on each particle, each with different parameters
2. measuring the mixing of each chain
3. resampling the parameter population based on adding noise to the parameters, weighted by the mixing score.
4. dropping the mutated particles, and taking a full SMC step with the new parameter population.

Due to the fact that the parameters are tuned before the SMC step is run, this is called **pretuning** approach.
```

## Define density and initial particles

```{code-cell} ipython3
n_particles = 5000
```

```{code-cell} ipython3
from jax.scipy.stats import multivariate_normal


def V(x):
    return 5 * jnp.sum(jnp.square(x**2 - 1))


def prior_log_prob(x):
    d = x.shape[0]
    return multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d))


loglikelihood = lambda x: -V(x)


def density():
    linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1)
    lambdas = jnp.linspace(0.0, 1.0, 5)
    prior_logvals = jnp.vectorize(prior_log_prob, signature="(d)->()")(linspace)
    potential_vals = jnp.vectorize(V, signature="(d)->()")(linspace)
    log_res = prior_logvals.reshape(1, -1) - jnp.expand_dims(
        lambdas, 1
    ) * potential_vals.reshape(1, -1)

    density = jnp.exp(log_res)
    normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
        linspace[1] - linspace[0]
    )
    density /= normalizing_factor
    return density
```

```{code-cell} ipython3
def initial_particles_multivariate_normal(dimensions, key, n_samples):
    return jax.random.multivariate_normal(
        key, jnp.zeros(dimensions), jnp.eye(dimensions) * 2, (n_samples,)
    )
```

# IRMH Kernel without pretunning, but with covariance matrix tuning (single parameter for all chains)

```{code-cell} ipython3
from blackjax import adaptive_tempered_smc
from blackjax.smc import resampling as resampling, solver, extend_params
from blackjax import irmh
from blackjax import inner_kernel_tuning
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_stds,
    particles_means,
)


def tuned_irmh_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.sampler_state.lmbda < 1

    def body(carry):
        i, state, op_key = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        return i + 1, state, op_key

    def f(initial_state, key):
        total_iter, final_state, _ = jax.lax.while_loop(
            cond, body, (0, initial_state, key)
        )
        return total_iter, final_state

    total_iter, final_state = f(initial_state, rng_key)
    return total_iter, final_state.sampler_state.particles
    
def irmh_full_cov_experiment(dimensions, target_ess, num_mcmc_steps):
    kernel = irmh.build_kernel()
    def step(key, state, logdensity, means, cov):
        "We need step to be vmappable over the parameter space, so we wrap it to make all parameter Jax Arrays or JaxTrees"
        proposal_distribution = lambda key: jax.random.multivariate_normal(
            key, means, cov
        )

        def proposal_logdensity_fn(proposal, state):
            return jnp.log(
                jax.scipy.stats.multivariate_normal.pdf(
                    state.position, mean=means, cov=cov
                )
            )

        return kernel(key, state, logdensity, proposal_distribution, proposal_logdensity_fn)
            

    def mcmc_parameter_update_fn(key, state, info):
        covariance = jnp.atleast_2d(particles_covariance_matrix(state.particles))
        return extend_params({"means":particles_means(state.particles), "cov":covariance})

    kernel_tuned_proposal = inner_kernel_tuning(
        logprior_fn=prior_log_prob,
        loglikelihood_fn=loglikelihood,
        mcmc_step_fn=step,
        mcmc_init_fn=irmh.init,
        resampling_fn=resampling.systematic,
        smc_algorithm=adaptive_tempered_smc,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=extend_params({"means":jnp.zeros(dimensions), "cov":jnp.eye(dimensions) * 2}),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    return kernel_tuned_proposal, tuned_irmh_loop
```

```{code-cell} ipython3
def smc_run_experiment(runnable, target_ess, num_mcmc_steps, dimen, key=rng_key):
    key, initial_particles_key, iterations_key = jax.random.split(key, 3)
    initial_particles = initial_particles_multivariate_normal(
        dimen, initial_particles_key, n_particles
    )
    kernel, inference_loop = runnable(dimen, target_ess, num_mcmc_steps)
    _, particles = inference_loop(
        kernel.step, iterations_key, kernel.init(initial_particles)
    )
    return particles
```

```{code-cell} ipython3
dimensions_to_try = [10, 20, 30]
```

```{code-cell} ipython3
experiments = []
dimensions = []
particles = []
for dims in dimensions_to_try:
    for exp_id, experiment in (
        ("tune_full_cov", irmh_full_cov_experiment),
    ):
        experiment_particles = smc_run_experiment(experiment, 0.5, 20, dims)
        experiments.append(exp_id)
        dimensions.append(dims)
        particles.append(experiment_particles)
```

```{code-cell} ipython3
results = pd.DataFrame(
    {"experiment": experiments, "dimensions": dimensions, "particles": particles}
)
```

```{code-cell} ipython3
linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1).squeeze()


def plot(post, sampler, dimensions, ax):
    post = np.asarray(post)
    dimensions = post.shape[1]
    for dim in range(dimensions):
        az.plot_kde(post[:, dim], ax=ax)
        _ = ax.plot(linspace, density()[-1], c="red")
rows = len(dimensions_to_try)
cols = 3
samplers = ["tune_full_cov",]
fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 5))

plt.rcParams.update({"font.size": 22})

for ax, lab in zip(axs[:, 0], dimensions_to_try):
    ax.set(ylabel=f"Dimensions = {lab}")

for ax, lab in zip(axs[0, :], samplers):
    ax.set(title=lab)

for col, experiment in enumerate(samplers):
    for row, dimension in enumerate(dimensions_to_try):
        particles = (
            results[
                (results.experiment == experiment) & (results.dimensions == dimension)
            ]
            .iloc[0]
            .particles
        )
        plot(particles, experiment, dimension, axs[row, col])

fig.tight_layout()
fig.suptitle(
    """Sampler comparison for increasing number of posterior dimensions.
Each plot displays all dimensions from the posterior, overlayed. The red curve is the actual marginal distribution."""
)
plt.show()
```

# Pretuning

```{raw-cell}
HMC has three parameters: inverse mass matrix, step size and number of integration steps.
We are going to keep the matrix constant, and hold a probability distribution the other two. The initial population
is sampled uniformly for reasonable bounds.
Each update step requires us to set sigma_parameters, a dictionary like:
```

````{raw-cell}
```python
    sigma_parameters={"step_size": 0.5, "num_integration_steps": 2},
```
````

````{raw-cell}
which specifies the magnitude of the noise used to explore the parameter space.

Moreover, since HMC requires number of steps to be an integer, we specify so by setting:

```python
round_to_integer=["num_integration_steps"]
```
````

```{code-cell} ipython3
import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.smc import extend_params, resampling
from blackjax.smc.pretuning import build_pretune, esjd, update_parameter_distribution

def pretuning_experiment(dimensions, target_ess, num_mcmc_steps):
    key = jax.random.PRNGKey(50)
    dimen=dimensions
    key, initial_particles_key, iterations_key = jax.random.split(key, 3)
    initial_particles = initial_particles_multivariate_normal(
            dimen, initial_particles_key, n_particles
    )
    sampling_key, step_size_key, integration_steps_key = jax.random.split(
        key, 3
    )

    # Set initial samples for integration steps and step sizes.
    integration_steps_distribution = jnp.round(
        jax.random.uniform(
            integration_steps_key, (n_particles,), minval=1, maxval=100
        )
    ).astype(int)
    
    step_sizes_distribution = jax.random.uniform(
        step_size_key, (n_particles,), minval=1e-2, maxval=1e-1
    )
    
    # Fixes inverse_mass_matrix and distribution for the other two parameters.
    initial_parameters = dict(
        inverse_mass_matrix=extend_params(jnp.eye(dimen)),
        step_size=step_sizes_distribution,
        num_integration_steps=integration_steps_distribution,
    )
    
    # As many parameters samples as particles
    assert initial_parameters["step_size"].shape == (n_particles,)
    assert initial_parameters["num_integration_steps"].shape == (n_particles,)

    # Pretuning step.
    pretune = build_pretune(
        blackjax.hmc.init,
        blackjax.hmc.build_kernel(),
        alpha=5,
        n_particles=n_particles,
        sigma_parameters={"step_size": 0.1, "num_integration_steps": 2},
        parameters_to_pretune=["step_size", "num_integration_steps"],
        round_to_integer=["num_integration_steps"],
    )

    # SMC definition
    init2, smc_kernel = blackjax.smc.pretuning.build_kernel(blackjax.tempered_smc,
        prior_log_prob,
        loglikelihood,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        resampling.systematic,
        num_mcmc_steps=num_mcmc_steps,
        pretune_fn=pretune)
    
    # Loop
    def body_fn(carry, lmbda):
        i, state = carry
        subkey = jax.random.fold_in(iterations_key, i)
        new_state, info = smc_kernel(subkey, state, lmbda=lmbda)
        return (i + 1, new_state), (new_state, info)
    
    a = init2(blackjax.tempered_smc.init, initial_particles, initial_parameters)

    # Run sampling
    num_tempering_steps = 20
    lambda_schedule = np.logspace(-5, 0, num_tempering_steps)
    (_, result), _ = jax.lax.scan(body_fn, (0, a), lambda_schedule)
    
    return result.sampler_state.particles, result   
    
```

# Plotting code

```{code-cell} ipython3
linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1).squeeze()
cols=3
rows=3
def plot(post, sampler, dimensions, ax):
    post = np.asarray(post)
    dimensions = post.shape[1]
    for dim in range(dimensions):
        az.plot_kde(post[:, dim], ax=ax)
        _ = ax.plot(linspace, density()[-1], c="red")
```

```{code-cell} ipython3
result1, final_state_1 = pretuning_experiment(10, 0.5, 20)
```

```{code-cell} ipython3
result2, final_state_2 = pretuning_experiment(20, 0.5, 20)
```

```{code-cell} ipython3
result3, final_state_3 = pretuning_experiment(30, 0.5, 20)
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(cols * 10, rows * 5))
plot(result1, "pretuning", 10, axs[0,0])
plot(result2, "pretuning", 10, axs[0,1])
plot(result3, "pretuning", 10, axs[1,0])
```

```{code-cell} ipython3
az.plot_kde(final_state_1.parameter_override["num_integration_steps"])
```

```{code-cell} ipython3
plt.scatter(final_state_1.parameter_override["num_integration_steps"], final_state_1.parameter_override["step_size"])
```

```{code-cell} ipython3
az.plot_kde(final_state_2.parameter_override["num_integration_steps"])
```

```{code-cell} ipython3
az.plot_kde(final_state_3.parameter_override["step_size"])
```

```{code-cell} ipython3
az.plot_kde(final_state_2.parameter_override["step_size"])
```

```{code-cell} ipython3
az.plot_kde(final_state_1.parameter_override["step_size"])
```

```{code-cell} ipython3
final_state_1.parameter_override["step_size"]
```

# Adaptive tempered

```{code-cell} ipython3
def pretuning_experiment_adaptive(dimensions, target_ess, num_mcmc_steps):
    key = jax.random.PRNGKey(50)
    dimen=dimensions
    key, initial_particles_key, iterations_key = jax.random.split(key, 3)
    initial_particles = initial_particles_multivariate_normal(
            dimen, initial_particles_key, n_particles
    )
    sampling_key, step_size_key, integration_steps_key = jax.random.split(
        key, 3
    )

    # Set initial samples for integration steps and step sizes.
    integration_steps_distribution = jnp.round(
        jax.random.uniform(
            integration_steps_key, (n_particles,), minval=1, maxval=50
        )
    ).astype(int)
    
    step_sizes_distribution = jax.random.uniform(
        step_size_key, (n_particles,), minval=1e-2, maxval=1e-1
    )
    
    # Fixes inverse_mass_matrix and distribution for the other two parameters.
    initial_parameters = dict(
        inverse_mass_matrix=extend_params(jnp.eye(dimen)),
        step_size=step_sizes_distribution,
        num_integration_steps=integration_steps_distribution,
    )
    
    # As many parameters samples as particles
    assert initial_parameters["step_size"].shape == (n_particles,)
    assert initial_parameters["num_integration_steps"].shape == (n_particles,)

    # Pretuning step.
    pretune = build_pretune(
        blackjax.hmc.init,
        blackjax.hmc.build_kernel(),
        alpha=5,
        n_particles=n_particles,
        sigma_parameters={"step_size": 0.1, "num_integration_steps": 2},
        parameters_to_pretune=["step_size", "num_integration_steps"],
        round_to_integer=["num_integration_steps"],
    )

    # SMC definition
    init2, smc_kernel = blackjax.smc.pretuning.build_kernel(blackjax.adaptive_tempered_smc,
        prior_log_prob,
        loglikelihood,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        resampling.systematic,
        num_mcmc_steps=num_mcmc_steps,
        pretune_fn=pretune,
        target_ess=target_ess)
    
    # Loop
    
    a = init2(blackjax.adaptive_tempered_smc.init, initial_particles, initial_parameters)

    # Run sampling
    key, initial_particles_key, iterations_key = jax.random.split(key, 3)
    initial_particles = initial_particles_multivariate_normal(
        dimen, initial_particles_key, n_particles
    )
    
    _, particles = tuned_irmh_loop(
        smc_kernel, iterations_key, a
    )

    
    return particles   
    
```

```{code-cell} ipython3
particles1 = pretuning_experiment_adaptive(10, 0.54, 10)
particles2 = pretuning_experiment_adaptive(20, 0.54, 10)
particles3 = pretuning_experiment_adaptive(30, 0.54, 10)
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(cols * 10, rows * 5))
plot(particles1, "pretuning_adaptive", 10, axs[0,0])
plot(particles2, "pretuning_adaptive", 20 , axs[0,1])
plot(particles3, "pretuning_adaptive", 30, axs[1,0])
```

```{code-cell} ipython3

```
