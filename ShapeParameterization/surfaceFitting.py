import os, sys, re
import numpy as np
np.float, np.int = float, int
import pyvista as pv
from scipy.interpolate import interp1d, griddata, RBFInterpolator
from geomdl import BSpline, utilities, exchange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname('FileRW'))
sys.path.append(os.path.dirname('ConvertFileType'))
sys.path.append(os.path.dirname('MeshGeneration'))
from FileRW.saveDatFile import *

def farthest_point_sampling(points, n):
        points = np.asarray(points)
        sampled = [np.random.randint(len(points))]  # start with a random point
        distances = np.full(len(points), np.inf)

        for _ in range(1, n):
            last_point = points[sampled[-1]]
            dist = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dist)
            next_idx = np.argmax(distances)
            sampled.append(next_idx)

        return points[sampled]

def evaluate_r2(x, y, z, surface):
    # Compute R2 value for fitting
    evalpts = np.array(surface.evalpts)
    z_pred = griddata((evalpts[:,0], evalpts[:,1]), evalpts[:,2], (x, y), method='cubic')

    # Remove any NaNs that happen during interpolation
    mask = ~np.isnan(z_pred)
    z_true = z[mask]
    z_pred = z_pred[mask]

    # Compute R2
    ss_res = np.sum((z_true - z_pred)**2)
    ss_tot = np.sum((z_true - np.mean(z_true))**2)
    r2_score = 1 - ss_res/ss_tot
    
    return r2_score

def fitSurfaceNURBs(pts):    
    from geomdl.visualization import VisVTK as vis
    from matplotlib import cm
    
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    dy_tol = 0.5
    deg_range = [2, 3, 4]

    # Structured sampling grid
    nx, ny = 30, 30
    y_bins = np.linspace(np.min(y), np.max(y), ny + 1)
    
    ctrlpts2d = []
    for i in range(ny):
        mask = (y >= y_bins[i]) & (y < y_bins[i+1])
        x_sub = x[mask]
        z_sub = z[mask]

        if len(x_sub) < 5:
            continue

        # Sort points by x
        idx_sort = np.argsort(x_sub)
        x_sub = x_sub[idx_sort]
        z_sub = z_sub[idx_sort]

        # Build interpolator
        interp_func = interp1d(x_sub, z_sub, kind='linear', fill_value="extrapolate")

        # Sample x between min and max where real points exist
        x_fit = np.linspace(x_sub.min(), x_sub.max(), nx)
        z_fit = interp_func(x_fit)

        row = [(xi, y_bins[i], zi) for xi, zi in zip(x_fit, z_fit)]
        ctrlpts2d.append(row)
        

    # Flatten the list for fitting
    flat_ctrlpts = [pt for row in ctrlpts2d for pt in row]
    size_u = len(ctrlpts2d)
    size_v = len(ctrlpts2d[0])
    
    best_r2 = -np.inf
    best_surface = None
    best_degrees = (None, None)

    for deg_u in deg_range:
        for deg_v in deg_range:
            try:
                # Build Surface
                surf = BSpline.Surface()
                surf.degree_u = deg_u
                surf.degree_v = deg_v
                surf.set_ctrlpts(flat_ctrlpts, size_u, size_v)

                surf.knotvector_u = utilities.generate_knot_vector(deg_u, size_u)
                surf.knotvector_v = utilities.generate_knot_vector(deg_v, size_v)
                
                r2 = evaluate_r2(x, y, z, surf)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_surface = surf
                    best_degrees = (deg_u, deg_v)
                    
            except Exception as e:
                print(f"Failed for degree ({deg_u},{deg_v}): {e}")

    surf.evaluate()
    evalpts = np.array(surf.evalpts)
    ctrlpts = np.array(surf.ctrlpts)

    return surf

def plotNurbsSurface(surf, pts):
    """
    Plots the fitted NURBs surface using matplotlib.
    """
    evalpts = np.array(surf.evalpts)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(evalpts[:, 0], evalpts[:, 1], evalpts[:, 2], c="k", s=2, alpha=0.6, label='NURBs Surface')
    ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], cmap='viridis', edgecolor='none', alpha=0.2, label=f"Mesh Surface")
    ax.set_title("Evaluated NURBs Surface")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def runSurfaceFitting(path, output_dir):
    
    mesh = pv.read(path)
    mesh = mesh.clean(tolerance=1e-8)  # Removes duplicate/close points
    points = mesh.points
    points_centered = points - points.mean(axis=0)

    # Step 1: PCA projection to UV space
    pca = PCA(n_components=2)
    uv = pca.fit_transform(points_centered)

    # Step 2: Convex hull in 2D for domain masking
    hull = ConvexHull(uv)
    hull_path = Path(uv[hull.vertices])

    # Step 3: Generate grid over bounding box
    grid_u, grid_v = 30, 30
    u_vals = np.linspace(uv[:, 0].min(), uv[:, 0].max(), grid_u)
    v_vals = np.linspace(uv[:, 1].min(), uv[:, 1].max(), grid_v)
    uu, vv = np.meshgrid(u_vals, v_vals)
    full_uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)

    # Step 4: Mask grid points outside convex hull
    valid_mask = hull_path.contains_points(full_uv)
    masked_uv = full_uv[valid_mask]

    # Step 5: RBF interpolation from UV to XYZ
    rbf = RBFInterpolator(uv, points)
    interpolated_xyz = rbf(masked_uv)

    saveControlPoints(interpolated_xyz, output_dir)

    # Step 6: Fill entire grid using scipy.griddata
    full_xyz = np.full((grid_u * grid_v, 3), np.nan)
    full_xyz[valid_mask] = interpolated_xyz

    valid_uv = full_uv[valid_mask]
    valid_xyz = full_xyz[valid_mask]

    filled_xyz = griddata(valid_uv, valid_xyz, full_uv, method='nearest')
    structured_xyz = filled_xyz.reshape((grid_v, grid_u, 3))

    # Step 7: Flatten to list of (x, y, z) for NURBs fitting
    if np.isnan(structured_xyz).any():
        raise ValueError("Interpolated surface grid contains NaNs — check point cloud coverage.")

    grid_pts = structured_xyz.reshape(-1, 3)

    # Step 8: Fit NURBs surface
    #nurbs_surf = fitSurfaceNURBs(grid_pts)
    nurbs_surf = fitSurfaceNURBs(points)

    # Step 9: Optional visualization
    #plotNurbsSurface(nurbs_surf, points)

    # Step 10: Export as IGES
    filename = os.path.basename(path)
    '''match = re.search(r"selected_surface_(\d+)", filename)
    if match:
        idx = int(match.group(1))
    save_path = os.path.join(output_dir, f"out_geom_{idx}.step")'''
    #save_path = os.path.join(output_dir, f"out_geom.step")
    #export_nurbs_surface_to_step(nurbs_surf, save_path, format='step')

    return points, interpolated_xyz, nurbs_surf

'''def selectControlNodes(path, output_dir, num_control_nodes=20):
    
    def farthest_point_sampling(points, n):
        points = np.asarray(points)
        sampled = [np.random.randint(len(points))]  # start with a random point
        distances = np.full(len(points), np.inf)

        for _ in range(1, n):
            last_point = points[sampled[-1]]
            dist = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dist)
            next_idx = np.argmax(distances)
            sampled.append(next_idx)

        return points[sampled]
    
    mesh = pv.read(path)
    mesh = mesh.clean(tolerance=1e-8)  # Removes duplicate/close points
    points = mesh.points
    
    controlNodes = farthest_point_sampling(points, num_control_nodes)

    return points, controlNodes'''

def selectControlNodes(path, output_dir, num_control_nodes=20):

    def farthest_point_sampling(points, n):
        print("Control Nodes selected using farthest point.")
        points = np.asarray(points)
        sampled = [np.random.randint(len(points))]
        distances = np.full(len(points), np.inf)
        for _ in range(1, n):
            last_point = points[sampled[-1]]
            dist = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dist)
            next_idx = np.argmax(distances)
            sampled.append(next_idx)
        return points[sampled]

    def bump_center_from_plane_residual(points: np.ndarray,
                                        frac: float = 0.03,
                                        min_pts: int = 30) -> np.ndarray:
        """
        Estimate bump center as weighted centroid of the most out-of-plane points.
        - frac: fraction of points considered 'bump' (top |d|)
        """
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 10:
            return pts.mean(axis=0)

        c = pts.mean(axis=0)
        X = pts - c

        # PCA plane normal = smallest singular vector
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        n = vt[-1]
        n /= (np.linalg.norm(n) + 1e-15)

        d = X @ n
        ad = np.abs(d)

        # choose top frac points by |distance|
        k = max(int(frac * len(pts)), min_pts)
        k = min(k, len(pts))
        idx = np.argpartition(ad, -k)[-k:]

        bump_pts = pts[idx]
        w = ad[idx]
        w = w / (w.sum() + 1e-15)

        center = (bump_pts * w[:, None]).sum(axis=0)
        return center

    def nearest_point(points: np.ndarray, target: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        t = np.asarray(target, dtype=float).reshape(1, 3)
        # fast enough; if huge, use cKDTree
        i = int(np.argmin(np.linalg.norm(pts - t, axis=1)))
        return pts[i:i+1]


    mesh = pv.read(path)
    mesh = mesh.clean(tolerance=1e-8)
    points = mesh.points

    if int(num_control_nodes) == 1:
        ctr = bump_center_from_plane_residual(points, frac=0.03, min_pts=30)
        controlNodes = nearest_point(points, ctr)
    else:
        controlNodes = farthest_point_sampling(points, num_control_nodes)

    return points, controlNodes


def farthest_point_sampling_indices(points, n, start_idx=None, seed=0):
    """
    Farthest-point sampling that returns point indices instead of coordinates.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Candidate point cloud.
    n : int
        Number of points to sample.
    start_idx : int or None
        Optional starting point index. If None, a deterministic random start is used.
    seed : int
        Seed used only when start_idx is None.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3). Got {pts.shape}")

    N = int(len(pts))
    if N == 0:
        raise ValueError("Cannot sample from an empty point cloud.")

    n = int(max(1, min(n, N)))

    if start_idx is None:
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, N))
    else:
        start_idx = int(np.clip(start_idx, 0, N - 1))

    sampled = [start_idx]
    distances = np.full(N, np.inf, dtype=float)

    for _ in range(1, n):
        last_point = pts[sampled[-1]]
        dist = np.linalg.norm(pts - last_point, axis=1)
        distances = np.minimum(distances, distances * 0 + dist)
        next_idx = int(np.argmax(distances))
        sampled.append(next_idx)

    return np.asarray(sampled, dtype=int)


def _surface_uv_coordinates(points):
    """
    Project a 3D surface point cloud to a stable 2D PCA coordinate system.
    Used only for region splitting/visual grouping, not for morphing.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3). Got {pts.shape}")

    if len(pts) < 3:
        return np.zeros((len(pts), 2), dtype=float), None

    pca = PCA(n_components=2)
    uv = pca.fit_transform(pts - pts.mean(axis=0, keepdims=True))
    return uv, pca


def split_surface_regions(points, n_regions=1, seed=0):
    """
    Split T-surface points into approximately spatial regions.

    The split is performed in a 2D PCA projection of the T-surface point cloud,
    then each 3D point is assigned a region id. For n_regions=1, all points are
    assigned to region 0.

    Returns
    -------
    point_region_ids : ndarray, shape (N,)
        Region id for every T-surface point.
    region_centres : ndarray, shape (R, 3)
        Representative 3D centre of each region, snapped to the nearest mesh node.
    centre_point_indices : ndarray, shape (R,)
        Point index of each representative centre in the original point cloud.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3). Got {pts.shape}")

    N = int(len(pts))
    if N == 0:
        raise ValueError("Cannot split an empty point cloud into regions.")

    R = int(max(1, min(n_regions, N)))

    if R == 1:
        point_region_ids = np.zeros(N, dtype=int)
        centre_point_indices = np.asarray([int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))], dtype=int)
        region_centres = pts[centre_point_indices]
        return point_region_ids, region_centres, centre_point_indices

    uv, _ = _surface_uv_coordinates(pts)

    # Prefer sklearn KMeans if available. It is deterministic with random_state.
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=R, random_state=int(seed), n_init=10)
        point_region_ids = km.fit_predict(uv).astype(int)
        centres_uv = km.cluster_centers_
    except Exception:
        # Fallback: farthest-point centres in UV + nearest-centre assignment.
        centre_uv_idx = farthest_point_sampling_indices(
            np.column_stack([uv, np.zeros(len(uv))]), R, seed=seed
        )
        centres_uv = uv[centre_uv_idx]
        d = np.linalg.norm(uv[:, None, :] - centres_uv[None, :, :], axis=2)
        point_region_ids = np.argmin(d, axis=1).astype(int)

    # Snap each region centre to the nearest original mesh point.
    tree = cKDTree(pts)
    centre_point_indices = []
    region_centres = []
    for rid in range(R):
        mask = point_region_ids == rid
        if np.any(mask):
            target = pts[mask].mean(axis=0)
        else:
            # Very rare, but handle empty clusters robustly.
            target_uv = centres_uv[rid]
            i_uv = int(np.argmin(np.linalg.norm(uv - target_uv, axis=1)))
            target = pts[i_uv]

        _, idx = tree.query(target, k=1)
        centre_point_indices.append(int(idx))
        region_centres.append(pts[int(idx)])

    return (
        point_region_ids.astype(int),
        np.asarray(region_centres, dtype=float),
        np.asarray(centre_point_indices, dtype=int),
    )


def selectRegionControlNodes(path, output_dir=None, n_regions=1, seed=0):
    """
    Select one representative control node per preliminary surface region.

    This is intended for the preliminary response-surface screening stage:
      - R regions are created on the T surface.
      - one centre control node is selected per region.
      - all T-surface points and selected control nodes receive region ids for plotting.

    Returns
    -------
    points : ndarray, shape (N, 3)
        T-surface points.
    control_nodes : ndarray, shape (R, 3)
        One representative centre node per region.
    point_region_ids : ndarray, shape (N,)
        Region id for every T-surface point.
    control_node_region_ids : ndarray, shape (R,)
        Region id for every selected control node.
    region_centres : ndarray, shape (R, 3)
        Same as control_nodes for this preliminary stage.
    control_node_point_indices : ndarray, shape (R,)
        Index of each selected CN in points.
    """
    mesh = pv.read(path)
    mesh = mesh.clean(tolerance=1e-8)
    points = np.asarray(mesh.points, dtype=float)

    point_region_ids, region_centres, centre_point_indices = split_surface_regions(
        points,
        n_regions=int(n_regions),
        seed=int(seed),
    )

    control_nodes = np.asarray(region_centres, dtype=float)
    control_node_region_ids = np.arange(len(control_nodes), dtype=int)

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "prelim_region_point_ids.npy"), point_region_ids)
            np.save(os.path.join(output_dir, "prelim_region_centres.npy"), region_centres)
            np.save(os.path.join(output_dir, "prelim_region_centre_indices.npy"), centre_point_indices)
        except Exception:
            pass

    return (
        points,
        control_nodes,
        point_region_ids,
        control_node_region_ids,
        region_centres,
        centre_point_indices,
    )
