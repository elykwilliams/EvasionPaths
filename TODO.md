# Bugs

- Bug in time-stepping when attempting to use topology_stack

# Style

- Use consistent types

# Features:

- Add examples of custom_sensor_network (two different motion models)
- Add example of custom_domain [In Progress]
- Add optional setting to enforce connected graph vs power-off
- integrate support for dynamical systems packages (pydstool, pydy, DynamicalBilliards.jl)
- Add sensor_network_tools (e.g. mixing, kinetic energy, etc)
- support time-series data (pandas) [In Progress]
- Implement 3D [In Progress]
- Maybe refactor sensor network into separate package
- Refactor Sensor Network from time-stepping into data stream.
- Refactor AlphaComplex decomposition and reeb graph generation

# Documentation:

- Migrate to more pythong based documentation
- Setup documentation git action
- Document Examples and tutorials
- Add github wiki
- use google format for src documentation
- Document refactored code
- Use type hints in all functions

# Testing

- Better error Messages
- Add logging
- Setup git action Continuous integration
- Setup full system tests
- Setup regression tests

# Structural:

- remove requirements.txt once pycharm supports pyproject.toml
- Use config file for example parameters

# Future Development

- Decaying sensing
- "Smart" Motion Model
- Work towards fully distributed algorithm
- Allow for fully disconnected graph ("power on")
- AlphaComplex with variable radii
- More applications
- IRL Sensor Network

