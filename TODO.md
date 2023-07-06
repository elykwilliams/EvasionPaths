# Bugs


# Style

- Use consistent types

# Features:

- Add examples of custom_sensor_network (two different motion models)
- Add optional setting to enforce connected graph vs power-off
- integrate support for dynamical systems packages (pydstool, pydy, DynamicalBilliards.jl)
- Add sensor_tools (e.g. mixing, kinetic energy, etc)
- support time-series data (pandas) [In Progress]
- Implement 3D [In Progress]
- Allow user to specify boundary shape/reflector. eg. allow for semi-circle
- Maybe refactor sensor network into separate package
- Refactor AlphaComplex decomposition and reeb graph generation

# Documentation:

- Migrate to pdoc for documentation
- Setup documentation git action
- Document Examples and tutorials
- src documentation move to google format
- Document refactored code
- Add type hints to all functions

# Testing

- Better error Messages
- Add logging
- Setup git action Continuous integration
- Setup full system tests
- Setup regression tests

# Structural:

- Install using pip/venv
- Update python version
- Use config file for parameters

# Future Development

- Decaying sensing
- "Smart" Motion Model
- Work towards fully distributed algorithm
- Measure Complexity and benchmark algorithm
- Allow for fully disconnected graph ("power on")
- AlphaComplex with variable radii

