import numpy as np
from pyquaternion import Quaternion

def tum_to_poses(tum_path):
    poses = []
    with open(tum_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = list(map(float, line.strip().split()))
            timestamp, x, y, z, qx, qy, qz, qw = parts
            q = Quaternion(qw, qx, qy, qz)
            poses.append((timestamp, np.array([x, y, z]), q))
    return poses

def calculate_delta_poses(poses):
    deltas = []
    for i in range(1, len(poses)):
        dt = poses[i][0] - poses[i-1][0]
        delta_translation = poses[i][1] - poses[i-1][1]
        delta_rotation = poses[i-1][2].inverse * poses[i][2]
        yaw, pitch, roll = delta_rotation.yaw_pitch_roll
        deltas.append(np.concatenate([delta_translation, [roll, pitch, yaw]]))
    return deltas