# Features:
 - Collective Motion Model
 - use elastic reflection in all cases
 - motion model should take in n_int_sensors, not n_total_sensors
 
 
# Bugs:
 - Points are not reflected during interpolation
 - Bug in graphing when all sensors become disconnected
 - Catch error when sensor "jumps the fence"?
 - With R = 0.2, N = 8, dt = 0.01, there were an abnormally large number of 
    
    `('Max recursion depth exceeded:(1, 1, 1, 1, 2, 2)', False)`
   
    `('Max recursion depth exceeded:(1, 1, 1, 1, 3, 3)', False)`
    
    This should be investigated as a potential bug

 
# Documentation:
 - Write MainPage
 - Document Examples
    
# Structural:
 - Add unit tests
 - refactor 1,2-simplex pair check into TopologicalState
 - refactor reflection??
 - add optional setting to enforce connected graph vs power-off
 
 # Experiments
 - N = 20, R = 0.18, RunAndTumble
 - More refined N/R sweep
 - N/R sweep with brownian motion
 - understand brownian motion?
 
 # Future Development:
 - Allow for fully disconnected graph ("power on")
 - Compute alpha complex fully in parallel
 - AlphaComplex with variable radii
 - Decaying sensing
 - Work towards fully distributed algorithm
 - 3D?
 - "Smart" Motion Model
 - Measure Complexity and benchmark algorithm
 
