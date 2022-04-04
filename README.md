# Emergent Periodicity in Fireflies
# Agent-Based Model

## Overview:
Individual fireflies show no periodicity in their flash pattern emission, drawing instead from a wide distribution of inter-flash interval timings to produce flashes. When in a collective, however, fireflies influence each other to change their flash timings, eventually reaching a steady state of synchronicity and, impressively, periodicity in their group level flash patterns. The simulation found here implements the theory presented in https://www.biorxiv.org/content/10.1101/2022.03.09.483608v1.full suggesting their interactions may follow a simple integrate-and-fire framework where flashes excite neighbors to flash faster than they would normally.

## Main requirements (versions tested on):
- Python 3.7
- NumPy 1.22.1
- scipy 1.17.1
- pandas 1.3.3
- Matplotlib 3.4.3

The complete list of required packages provided in *requirements.txt*, which you can install in your environment with the command `pip install -r requirements.txt`. 

## Model usage:
Default parameters are already set up. To further set up or change parameters for the simulation, a number of flags can be set. Free parameters of the model discussed in the paper:
- `n`: Number of fireflies in arena (default: `20`)
- `beta': Donation provided by any one flash (default: `0.2`)

For a list of customizable flags, please run **'python run.py -h'

**`python run.py`** runs a simulation with the default parameters.

### Output:
In the folder *simulation_and_theory_results* there will be a pickle file named to match the density and beta parameters you provided. In here are dictionaries mapping each firefly to its respective a) flash time series (0s and 1s), b) voltage time series (continuous variable between 0 and 1), c) times of flashes (discrete values), d) positions at each itme step (discrete values), which can all be viewed by loading and unpickling the object.

Simulation.py and Firefly.py are objects used by run.py to instantiate and carry out the logic of the simulation. 

