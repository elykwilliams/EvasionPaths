# Evasion Paths

This repository implements the algorithm described in _[Evasion paths in mobile sensor networks](https://arxiv.org/pdf/1308.3536.pdf)_.

See our results _[Efficient Evader Detection in Mobile Sensor Networks](https://arxiv.org/abs/2101.09813)_ 
# System Requirements
This repository uses python modules that are easiest to install using [Anaconda](https://www.anaconda.com/).
Installation using pip/venv is being tested. 

# Clone Repository
To download this repository from the command line, use the following 

`git clone https://www.github.com/elykwilliams/evasion-paths.git`

# Configuration 
The `enviroment.yml` file specifies the exact python environment used; this ensures consistant results across platforms. 

To configure using the Makefile

`make install`


This will create the EvasionPaths-env conda environment. To run examples you will still need to activate and deactivate the environment.

To activate:
`conda activate EvasionPaths-env`

To deactivate:
`conda deactivate EvasionPaths-env`

# Configuation in Windows
Use the Anaconda Navigator to import the environment.yml file. This will create the environment. 
Then, be sure to set this environment as the project interpreter in your IDE of choice. 
Finally, be sure to set the `src` folder as a Source folder in your IDE (otherwise it will complain about modules not being found)

# Documentation
Documentation for this project can be found [here](https://elykwilliams.github.io/EvasionPaths/)



