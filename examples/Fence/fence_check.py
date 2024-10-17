import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from alpha_complex import AlphaComplex, Simplex

def UnitCubeFence(spacing):
    epsilon = 1e-5  # Perturbation factor
    dx = np.sqrt(3) * spacing / 2

    # Create a grid of points along x, y, and z coordinates
    points = np.arange(-dx, 1.001 + dx, spacing)
    grid = list(product(points, points))

    # Generate perturbed grid points for each face of the unit cube
    x0_face = [(-dx + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon), z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y0_face = [(x + random.uniform(-epsilon, epsilon), -dx + random.uniform(-epsilon, epsilon), z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z0_face = [(x + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon), -dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    x1_face = [(1 + dx + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon), z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y1_face = [(x + random.uniform(-epsilon, epsilon), 1 + random.uniform(-epsilon, epsilon), z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z1_face = [(x + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon), 1 + random.uniform(-epsilon, epsilon)) for x, y in grid]

    # Combine all face points and remove duplicates
    fence_sensors = np.concatenate((x0_face, y0_face, z0_face, x1_face, y1_face, z1_face))
    return np.unique(fence_sensors, axis=0)

def find_simplex_neighbors(sensors, spacing):
    """
    Finds two points that, along with Point 0, form a valid 2-simplex (triangle) within the given spacing.

    Parameters:
    sensors (np.ndarray): Array of sensor positions.
    spacing (float): The distance threshold to form a simplex.

    Returns:
    tuple: Indices of the points forming the simplex, or None if not found.
    """
    # Compute pairwise distances between Point 0 and all other points
    distances = cdist([sensors[0]], sensors[1:], metric='euclidean').flatten()

    # Identify points within the spacing distance from Point 0
    valid_neighbors = np.where(distances < spacing)[0] + 1  # +1 to correct for index offset due to slicing

    # Check pairs of valid neighbors to see if they also form a simplex with each other
    for i in range(len(valid_neighbors)):
        for j in range(i + 1, len(valid_neighbors)):
            p1, p2 = valid_neighbors[i], valid_neighbors[j]
            if np.linalg.norm(sensors[p1] - sensors[p2]) < spacing:
                return 0, p1, p2  # Return indices of the valid simplex points

    return None

def plot_sensors_3d(sensors, simplex_indices=None):
    """
    Plots the sensors in 3D space.

    Parameters:
    sensors (np.ndarray): Array of sensor positions.
    simplex_indices (tuple): Indices of the points forming the desired simplex (optional).
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all sensors in blue
    ax.scatter(sensors[:, 0], sensors[:, 1], sensors[:, 2], color='blue', label='Sensors')

    # Highlight points in the simplex in orange
    if simplex_indices:
        ax.scatter(sensors[simplex_indices, 0], sensors[simplex_indices, 1], sensors[simplex_indices, 2], color='orange', s=100, label='Simplex Points {0,1,2}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Sensor Fence with Highlighted Simplex')
    ax.legend()
    plt.show()

def reindex_for_simplex(alpha_complex, fence_points):
    # Traverse the 2-simplices in the alpha complex
    for simplex, _ in alpha_complex.simplex_tree.get_filtration():
        if len(simplex) == 3 and 0 in simplex:  # Find the first 2-simplice with vertex 0
            print(f"Found 2-simplex containing vertex 0: {simplex}")
            # Swap indices to ensure {0, 1, 2}
            indices = list(simplex)
            indices.remove(0)  # Remove 0 to get the other two indices
            idx1, idx2 = indices

            # Swap points so that they correspond to {0, 1, 2}
            new_fence = np.copy(fence_points)
            new_fence[[1, 2]] = fence_points[[idx1, idx2]]
            new_fence[[idx1, idx2]] = fence_points[[1, 2]]
            print(f"Swapped points: 1 <-> {idx1}, 2 <-> {idx2}")
            print("old fence: ")
            print(fence_points[0:3], fence_points[[idx1, idx2]])
            print("new fence: ")
            print(new_fence[0:3], new_fence[[idx1, idx2]])
            return True, new_fence

    # If no suitable simplex is found
    print("Could not find a 2-simplex containing vertex 0 in the alpha complex.")
    return False

def main():
    # Define the spacing for the fence grid
    spacing = 0.2

    # Generate sensor positions using the UnitCubeFence function
    sensors = UnitCubeFence(spacing)
    print(f"Total number of sensors generated: {len(sensors)}")
    alpha_complex = AlphaComplex(sensors, np.sqrt(spacing))

    reindex_success, reordered_sensors = reindex_for_simplex(alpha_complex, sensors)


    if reindex_success:
        # Check again by recreating the alpha complex
        alpha_complex = AlphaComplex(reordered_sensors, spacing)

        # Check if {0, 1, 2} is present as a 2-simplex
        target_simplex = Simplex({0, 1, 2})
        is_present = target_simplex in alpha_complex.simplices(2)
        print(alpha_complex.simplices(2))
        print(f"Checking if {target_simplex} is in the alpha complex simplices...")
        print(f"Simplex {target_simplex} found: {is_present}")

        # Plot the points with highlighted simplex
        plot_sensors_3d(reordered_sensors, simplex_indices=(0, 1, 2))
    else:
        print("No suitable simplex found.")

if __name__ == "__main__":
    main()

