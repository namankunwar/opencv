import numpy as np
from scipy.spatial.distance import cdist

def compute_cumulative_distances(points):
    """
    Compute cumulative distances between consecutive points in a list of points.
    """
    points = np.array(points)
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))  # Euclidean distances
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Cumulative sum of distances
    return cumulative_distances

def resample_points(points, num_points):
    """
    Resample points to get equally spaced points along the curve defined by the original points.
    """
    # Compute cumulative distances
    cumulative_distances = compute_cumulative_distances(points)
    
    # Determine the new equally spaced distances
    total_length = cumulative_distances[-1]
    target_distances = np.linspace(0, total_length, num_points)
    
    # Interpolate points at target distances
    new_points = []
    for target in target_distances:
        idx = np.searchsorted(cumulative_distances, target)
        if idx == 0:
            new_points.append(points[0])
        elif idx == len(cumulative_distances):
            new_points.append(points[-1])
        else:
            p1, p2 = np.array(points[idx - 1]), np.array(points[idx])  # Convert tuples to numpy arrays
            dist1, dist2 = cumulative_distances[idx - 1], cumulative_distances[idx]
            # Linear interpolation between p1 and p2
            weight = (target - dist1) / (dist2 - dist1)
            new_point = p1 + weight * (p2 - p1)
            new_points.append(new_point)
    
    return np.array(new_points)

# Example usage:
points = [(139.66900972815253, 150.17235772782365), (138.72802504663485, 150.6314576599205), 
          (137.78409092148743, 151.09623200328488), (136.83634480358654, 151.57390142712256),
          (135.88463976950294, 152.06328474243247), (134.92055845712846, 152.57824018888184),
          (133.94351231325444, 153.12011839096218), (132.93690872575752, 153.70076813069667),
          (131.90049371495323, 154.31773060176857), (130.8318118959615, 154.97346920546738)]

# Resample points to make them equidistant
num_points = 50  # You can adjust this based on how many equidistant points you need
equidistant_points = resample_points(points, num_points)

print("Equidistant Points: ", equidistant_points)
