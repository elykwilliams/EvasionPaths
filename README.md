# Evasion Paths

This repository implements the algorithm described in _[Evasion paths in mobile sensor networks](https://arxiv.org/pdf/1308.3536.pdf)_.

See our results _[Efficient Evader Detection in Mobile Sensor Networks](https://arxiv.org/abs/2101.09813)_ 

# System Requirements
This repository uses python modules that are easiest to install using [Anaconda](https://www.anaconda.com/).
Installation using pip/venv is being tested.

# Clone Repository
To download this repository from the command line, use the following

`git clone https://www.github.com/elykwilliams/evasion-paths.git`

# Pip Installation
To install using pip, simply use
`make install`

This will create a virtual environment called `env`. This environment will need to be loaded before running any scripts.

To activate:
`source env/bin/activate`

And to deactivate:
`deactivate`

To update the project, use
`make update`

This will pull the most recent updates from github and reinstall everything.

# Conda Installation
Conda installation is similar
`make conda-install`

This will create an environment called `EvasionPaths-env` which will need to be activated before running any scripts.

To activate:
`conda activate EvasionPaths-env`

To deactivate:
`conda deactivate EvasionPaths-env`

We can also update the project using
`make conda-update`

This will pull the most recent updates from github and reinstall everything.


# Configuation in Windows
Use the Anaconda Navigator to import the environment.yml file. This will create the environment. 
Then, be sure to set this environment as the project interpreter in your IDE of choice. 
Finally, be sure to set the `src` folder as a Source folder in your IDE (otherwise it will complain about modules not being found)

If you import the project using PyCharm, it will attempt to load the 

# Documentation
Documentation for this project can be found [here](https://elykwilliams.github.io/EvasionPaths/)



