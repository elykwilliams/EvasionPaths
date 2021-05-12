 
# Style
 - remove domain from sensor network initialization
 - Use consistent list/tuple/array
 - Simplify plotting tools 
 - viseck/DO don't need radius member, use from sensor_network
 - ODE motion doesn't need n_sensors
 - compute cmap alpha cycle from fence sensors

# Features:
 - Add sensor_tools
 - Add examples of custom_sensor_network
 - add optional setting to enforce connected graph vs power-off

# Documentation:
 - Write MainPage
 - Document Examples
    
# Testing
 - Better error Messages
 - Add logging
 - Add unit tests

# Structural:
 - Install using pip/venv 
 - config file?

 # Summer Development:
 - Allow for fully disconnected graph ("power on")
 - AlphaComplex with variable radii
 - Work towards fully distributed algorithm
 - 3D
 
# Future Development
 - Decaying sensing
 - "Smart" Motion Model
 - Measure Complexity and benchmark algorithm
