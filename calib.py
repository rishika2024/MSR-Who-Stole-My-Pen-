import numpy as np
from scipy.spatial.transform import Rotation as R

# Data
camera_points = np.array([
    [0.02234892, -0.0541107, 0.34890001],
    [-0.09026493, 0.03537184, 0.34936364],
    [-0.18827625, 0.00991136, 0.34972771]
])

robot_points = np.array([
    [0.09, 0., 0.08422],
    [0.196665639, -0.00150843621, 0.0491259130],
    [0.25183136, -0.00193156, 0.12078869]
])

# Compute centroids
camera_centroid = np.mean(camera_points, axis=0)
robot_centroid = np.mean(robot_points, axis=0)

# Compute centered vectors (differences from centroid)
camera_diff = camera_points - camera_centroid
robot_diff = robot_points - robot_centroid

# Align vectors
rot, rssd, sens = R.align_vectors(camera_diff, robot_diff, return_sensitivity=True)
rotation_matrix = rot.as_matrix()

# Compute translation
translation = robot_centroid - rot.apply(camera_centroid)

print("Camera centroid:", camera_centroid)
print("Robot centroid:", robot_centroid)
print("Rotation matrix:\n", rotation_matrix)
print("Translation vector:", translation)
