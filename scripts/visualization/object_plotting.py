import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scripts.utils.transforms import transform_points


def make_cube_vertices(size):
    s = size / 2.0
    return np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ], dtype=float)


def cube_faces_from_vertices(v):
    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[0], v[3], v[7], v[4]],  # left
    ]


def create_cube_artist(ax, alpha=0.25, facecolor="cyan", edgecolor="k"):
    poly = Poly3DCollection([], alpha=alpha, facecolor=facecolor, edgecolor=edgecolor)
    ax.add_collection3d(poly)
    return poly


def update_cube_mesh(poly, T_world_from_cube, cube_size):
    verts_local = make_cube_vertices(cube_size)
    verts_world = transform_points(T_world_from_cube, verts_local)
    faces = cube_faces_from_vertices(verts_world)
    poly.set_verts(faces)