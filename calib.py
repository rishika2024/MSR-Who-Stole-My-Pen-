import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_calibration():
    
    # Data
    camera_points = np.array([
        [ 0.07533889, -0.05585445,  0.32900003],
        [-0.01806629,  0.00711228,  0.38500002],
        [-0.10960154, -0.07563574,  0.42000002],
        [-0.12632586, -0.15719515,  0.41800001]

        
    ])

    robot_points = np.array([        
        [0.09057223, 0.0,  0.08396219],
        [0.19578836, 0.0,  0.04812197],
        [0.25173868, 0.0,  0.12048092],
        [0.25001417, 0.0,  0.16486356]

        
    ])

    # Compute centroids
    camera_centroid = np.mean(camera_points, axis=0)
    robot_centroid = np.mean(robot_points, axis=0)

    # Compute centered vectors
    camera_diff = camera_points - camera_centroid
    robot_diff = robot_points - robot_centroid

    # Align vectors
    rot, rssd, sens = R.align_vectors(camera_diff, robot_diff, return_sensitivity=True)
    rotation_matrix = rot.as_matrix()

    # Compute translation
    translation = robot_centroid - rot.apply(camera_centroid)

    print(f"rotation matrix = {rotation_matrix}, \n, translation_matrix={translation}")

    return rotation_matrix, translation

compute_calibration()