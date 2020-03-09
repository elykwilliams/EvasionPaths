# Features:
 - Billiard Motion
 - Run and Tumble Motion
 - Compress long animations
 - Add simplex.evasion_path to list per timestep to generate log
 
# Bugs:
 - Fix Environment files to be portable
 - Unknown Error somewhere in evasion_path.run(): 
 
        "list assignment index out of range"
 
    
# Development:
 - Work towards allowing disconnected graph
 - Compute alpha complex fully in parallel
 - AlphaComplex with variable radii
 - Work towards fully distributed algorithm
 - 3D?
 - Smart Motion Model?

# Documentation:
 - Finish writing documentation
 - Go through and comment files
 - Add Licence.txt
 - Measure Complexity and benchmark algorithm
 - DOxygen?
 
# Structural:
 - Incorporate virtual boundary into Boundary not MotionModel
 - Simplify example scripts
 - Is it possible to separate out find_evasion_paths() function from EvasionPath class?
 - ....Somthing like find_evasion_path(old_complex, new_complex)
 - Use dart representation of boundary cycles instead of of just a set of nodes
 - Rework Combinatorial Map using networkx.DiGraph?
 - ....or use dart representation of bcycles. Maybe manage dart mapping directly by comparing added/removed edges?
 - Refactor EvasionPathSimulation.points completely into MotionModel  
 - Move point generation inside boundary class
# Experiments
 - N - slice
 - R - slice
 - N-R sample
 