# Using Deep Generative Priors For Inverse ProblemsAnd Bayesian Optimisation

There is a [wiki](https://boiled-mat-c58.notion.site/Discriminator-Weighted-Sampling-Wiki-8cbfa778c79345f69352a0fb94bb6f4f) over here for more details. 

The basic struture of the YAML files inside config folder

```
exp_params:
  ... # Basic info that is constantly used by multiple functions, basically a dump right now

operator_params:
  ... # Info defining the operator

estimator_params:
  ... # This is where all the interesting things for the project happens actually

base_model_params:
  ... # This references an existing YAML that describes the base model
```

## operator_params 
Functions that are currently supported
- operator: CenterOcclude/RandomOcclude/CompressedSensing

## estimator_params 
Functions that are currently supported, these are the path to config files
- potential: mse/discriminator_weighted
- initalisation: random/posterior/map_posterior
- estimator: langevin/hmc/map/mala

## base_model_params 
Functions that are currently supported, these are the path to config files
- model_name: mnist/gan/dcgan or mnist/vae/vanilla
