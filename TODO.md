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
 - decaying radii?
# Documentation:
 - Finish writing documentation
 - Go through and comment files
 - Add Licence.txt
 - Measure Complexity and benchmark algorithm
 - DOxygen?
 
# Structural:
 - Incorporate virtual boundary into Boundary not MotionModel
 - move reflection into boundary class??
 - Move point generation inside boundary class??
 - Simplify example scripts
 - Is it possible to separate out find_evasion_paths() function from EvasionPath class?
 - ....Something like find_evasion_path(old_complex, new_complex)
  # Experiments
 - N - slice
 - R - slice
 - N-R sample
 