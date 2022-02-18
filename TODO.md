# Bugs

- Bug was introduced: sample_animation errors out for small radius after some time. Need to fix with better try catch
    - Not sure if bug present after refactor

# Style

- Use consistent list/tuple/array/set

# Features:

- Add examples of custom_sensor_network (two different motion models)
- Add optional setting to enforce connected graph vs power-off

- integrate support for dynamical systems packages (pydstool, pydy)
- Add sensor_tools (e.g. mixing, kinetic energy, etc)
- support time-series data (pandas)
- Implement 3D

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
- Refactor Domain/MotionModel/SensorNetwork

# Summer Development:
- Allow for fully disconnected graph ("power on")
- AlphaComplex with variable radii

# Future Development

- Decaying sensing
- "Smart" Motion Model
- Work towards fully distributed algorithm
- Measure Complexity and benchmark algorithm
