# Evasion Paths

This repository implements the algorithm described in _[Evasion paths in mobile sensor networks](https://arxiv.org/pdf/1308.3536.pdf)_.

# Installation

## Clone Repository
To download this repository, use the following 

`git clone https://www.github.com/elykwilliams/EvasionPaths.git`

## Configure Virtual Environment
To make sure that the installed packages to not interfer with system packages, we use a virtual environment
```
cd /path/to/EvasionPaths
python3 -m venv venv
```
To activate the virtual environment on Windows
`.\venv\Scripts\activate`

To activate the virtual environment on Linux
`source venv/bin/activate`

You can deactivate the virtual environment using `deactivate`

## Install package
After activating the virtual environment, the evasions path package can be installed using 
```
cd /path/to/EvasionPaths
pip install .
```

# Running Examples
After the package is installed to the virtual environment, the examples can be run using 
```
cd /path/to/EvasionPaths
python examples/sample_experiment.py
```

# Documentation
Documentation for this project can be found [here](https://elykwilliams.github.io/EvasionPaths/)


