import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CONTROL_NODES = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Old\surf\Control Nodes\control_nodes.npy"
CONTROL_NORMALS = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Old\surf\Control Nodes\control_normals.npy"

KNN = 3
AMP = 20.0
SEED = 7

rng = np.random.default_rng(SEED)

def load_data():
    cn = np.load(CONTROL_NODES).astype(float)
    normals = np.load(CONTROL_NORMALS).astype(float)

    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    return cn, normals


from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

def build_flat_surface_connectivity(points):
    """
    Build surface connectivity for approximately flat 3D control nodes.

    Steps:
    1. Fit best-fit plane using PCA/SVD.
    2. Project 3D points onto the 2D plane coordinates.
    3. Perform 2D Delaunay triangulation.
    4. Return triangular faces and edges using original point indices.
    """

    P = np.asarray(points, dtype=float)

    # Centre the point cloud
    centroid = P.mean(axis=0)
    Q = P - centroid

    # PCA plane basis
    # Vt[0], Vt[1] are in-plane directions
    # Vt[2] is the plane normal
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)

    e1 = Vt[0]
    e2 = Vt[1]

    # Project points into 2D plane coordinates
    uv = np.column_stack((Q @ e1, Q @ e2))

    # Delaunay triangulation in 2D
    tri = Delaunay(uv)
    faces = tri.simplices

    # Extract edges from triangles
    edges = set()
    for face in faces:
        i, j, k = map(int, face)
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    return faces, sorted(edges), uv


def build_surface(points):
    hull = ConvexHull(points)
    return hull.simplices


def set_equal_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centre = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)

    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def clean_axis(ax, title):
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_control_nodes(points):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=45)

    clean_axis(ax, "Control nodes")
    set_equal_3d(ax, points)

    plt.tight_layout()
    plt.show()


def plot_surface_connectivity(points, faces, edges):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    tris = points[faces]
    surf = Poly3DCollection(tris, alpha=0.18)
    ax.add_collection3d(surf)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=35)

    for i, j in edges:
        p = points[[i, j]]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=0.8)

    clean_axis(ax, f"Generated surface edges")
    set_equal_3d(ax, points)

    plt.tight_layout()
    plt.show()


def plot_morphed_surface(base, morphed, faces, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    base_surf = Poly3DCollection(base[faces], alpha=0.12)
    morph_surf = Poly3DCollection(morphed[faces], alpha=0.35)

    ax.add_collection3d(base_surf)
    ax.add_collection3d(morph_surf)

    ax.scatter(base[:, 0], base[:, 1], base[:, 2], s=20, alpha=0.35)
    ax.scatter(morphed[:, 0], morphed[:, 1], morphed[:, 2], s=35)

    for p0, p1 in zip(base, morphed):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linestyle="--",
            linewidth=0.7,
            alpha=0.7,
        )

    clean_axis(ax, title)
    set_equal_3d(ax, np.vstack([base, morphed]))

    plt.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(SEED)

    cn, normals = load_data()

    faces, edges, uv = build_flat_surface_connectivity(cn)

    # Random XYZ displacement
    d_xyz = rng.normal(0.0, 1.5, size=cn.shape)
    #d_xyz /= np.linalg.norm(d_xyz, axis=1, keepdims=True) + 1e-12
    d_xyz *= AMP
    cn_xyz = cn + d_xyz

    # Random normal displacementP
    scalar = rng.normal(0.0, 1.5, size=(len(cn), 1))
    d_normal = AMP * scalar * normals
    cn_normal = cn + d_normal

    plot_control_nodes(cn)
    plot_surface_connectivity(cn, faces, edges)
    plot_morphed_surface(cn, cn_xyz, faces, "Morphed surface: random XYZ displacement")
    plot_morphed_surface(cn, cn_normal, faces, "Morphed surface: random normal displacement")


if __name__ == "__main__":
    main()