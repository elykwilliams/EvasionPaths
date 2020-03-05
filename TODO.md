# Features:
 - Billiard Motion
 - Run and Tumble Motion
 - Compress long animations
 - Add color legend to animation
 
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
 - Refactor do_timestep() and do_adaptive_timestep() into one function
 - Simplify example scripts
 - Is it possible to separate out find_evasion_paths() function from EvasionPath class?
 - Use dart representation of boundary cycles instead of of just a set of nodes
 - Figure out confusing `sorted_edges[]` in Combinatorial Map 

# Experiments
 - R-slice
 - N-R sample
 