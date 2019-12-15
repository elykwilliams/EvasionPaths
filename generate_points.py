import numpy as np
import csv

filename = 'pointgenration.txt'

r = 0.25
n_interior_sensors = 30
with open(filename,'w') as file:
    csv_file = csv.writer(file, delimiter=",")

    for point in [[0,0], [0,1], [1,0], [1,1]]:
        csv_file.writerow(point)
    
    for i in np.arange(r, 1, r):
        csv_file.writerow([0, i])
        csv_file.writerow([1, i])
        csv_file.writerow([i, 0])
        csv_file.writerow([i, 1])
    
    for _ in range(n_interior_sensors):
        csv_file.writerow(np.random.uniform(r, 1-r, 2))
        
          
