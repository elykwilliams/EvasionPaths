# Evasion Paths

This repository implements the algorithm described in _[Evasion paths in mobile sensor networks](https://arxiv.org/pdf/1308.3536.pdf)_.

# System Requirements
This repository uses python modules that are easiest to install using [Anaconda](https://www.anaconda.com/)

# Clone Repository
To download this repository, use the following 

`git clone https://www.github.com/elykwilliams/evasion-paths.git`

# Configuration 
The `enviroment.yml` file specifies the exact python environment used; this ensures consistant results across platforms. 

To configure using the Makefile

`make install`


This will create the EvasionPaths-env conda environment. To run examples you will still need to activate and deactivate the environment.

To activate:
`conda activate EvasionPaths-env`

To deactivate:
`conda deactivate`

# Configuation in Windows
Use the Anaconda Navigator to import the environment.yml file. This will create the environment. 
Then, be sure to set this environment as the project interpreter in your IDE of choice. 
Finally, be sure to set the `src` folder as a Source folder in your IDE (otherwise it will complain about modules not being found)

# Documentation
Documentation for this project can be found [here](https://elykwilliams.github.io/EvasionPaths/)

# Example Files
To familiarize yourself with how to run the code, we include four example files in the examples directory. 

- sample_experiment.py
    - This will run a simulation in a basic manner, running the simulations sequentially and without descriptive exceptions. 

- parallel_experiments.py
    - This will run a batch of simulations in parallel to reduce run time. 

- sample_long_experiment.py
    - This will run a batch of simulations in parallel with descriptive exceptions raised if the simulation fails. 

- sample_animation.py
    - This will run a single simulation and produce an animation so that you can visualize different configurations of sensors and models of motion. 


There is also a file sample_run.sh included in the examples directory that can be used to run any of the above scripts. 

