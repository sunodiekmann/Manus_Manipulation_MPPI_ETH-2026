import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)

    qw, qx, qy, qz = pose[3:]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix

    Returns:
    numpy.ndarray: Quaternion in format [w, x, y, z]
    """
    # Ensure R is a valid rotation matrix
    if R.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3")

    # Check for proper orthogonal matrix
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError("Input matrix must have determinant 1")

    # Computing trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        # If trace is positive
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # If R[0,0] is the largest diagonal entry
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        # If R[1,1] is the largest diagonal entry
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        # If R[2,2] is the largest diagonal entry
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    # Return quaternion as [w, x, y, z]
    return np.array([w, x, y, z])


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_out = (T @ pts_h.T).T
    return pts_out[:, :3]

def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_out = (T @ pts_h.T).T
    return pts_out[:, :3]

def matrix_to_dict(T):
    rot = T[:3, :3]
    trans = T[:3, 3]
    quat = R.from_matrix(rot).as_quat()  # xyzw
    return {
        "matrix_4x4": T.tolist(),
        "translation_xyz": trans.tolist(),
        "quaternion_xyzw": quat.tolist(),
    }

def rigid_transform_3d(A, B):
    """
    Solve A -> B rigid transform for corresponding 3D points.

    A: (N,3) points in source frame
    B: (N,3) points in target frame

    Returns:
        R_mat: (3,3)
        t: (3,)
    such that:
        B ~= R_mat @ A + t
    """
    assert A.shape == B.shape
    assert A.shape[1] == 3
    assert A.shape[0] >= 3

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B
    
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T

    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t = centroid_B - R_mat @ centroid_A
    return R_mat, t

def build_transform(R_mat, t):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def safe_inverse_transform(T):
    """Return the inverse of a 4x4 rigid-body transform, or None if input is None."""
    if T is None:
        return None
    return np.linalg.inv(T)
    return T