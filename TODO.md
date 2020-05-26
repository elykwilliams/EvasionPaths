# Features:
 - Collective Motion Model
 
# Bugs:
 - Points are not reflected during interpolation
 - Bug in graphing when all sensors become disconnected
 - Catch error when sensor "jumps the fence"
 
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
 
 # Future Development:
 - Allow for fully disconnected graph ("power on")
 - Compute alpha complex fully in parallel
 - AlphaComplex with variable radii
 - Decaying sensing
 - Work towards fully distributed algorithm
 - 3D?
 - "Smart" Motion Model
 - Measure Complexity and benchmark algorithm
 
