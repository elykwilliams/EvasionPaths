# Features:
 - Collective Motion Model
 
# Bugs:
 - Fix Environment files to be portable
 - Unknown Error somewhere in evasion_path.run(): 
 
        "list assignment index out of range"
   Error may no longer exist

# Development:
 - Allow for fully disconnected graph
 - Compute alpha complex fully in parallel
 - AlphaComplex with variable radii
 - Work towards fully distributed algorithm
 - 3D?
 - Smart Motion Model?
 - decaying radii?
 
# Documentation:
 - Finish writing documentation
 - Go through and comment files
 - Add Licence.txt
 - Measure Complexity and benchmark algorithm
 - DOxygen?
 
# Structural:
 - Simplify example scripts
 - add more example scripts
 - Clean up n_sensors interface
 - Clean up cycle labelling interface
 - refactor update_labeling()??
 - Simplify Delauney flip check logic 
 - bounday should have alpha-cycle
 - Add unit tests
 
 # Experiments
 - N - slice
 - R - slice
 - N-R sample
 