import os, sys
from scipy.spatial import cKDTree
import networkx as nx
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import pyvista as pv
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from geomdl import utilities, BSpline
from geomdl.visualization import VisMPL

sys.path.append(os.path.dirname('MeshGeneration'))
sys.path.append(os.path.dirname('ConvertFileType'))
from MeshGeneration.meshFile import *
from ConvertFileType.convertToStep import *


def build_nurbs_surface_from_splines(spline_list, degree_u=3, degree_v=3):
    """
    Builds a NURBS surface from a list of 3D spline curves using direct control point assignment.
    """
    curves = np.array(spline_list)  # shape: [n_curves, n_points, 3]
    if curves.ndim != 3 or curves.shape[2] != 3:
        raise ValueError("Each spline must be a (N, 3) array")

    n_curves = curves.shape[0]
    n_points = curves.shape[1]

    # Transpose to match [chordwise][spanwise][3]
    ctrlpts_grid = np.transpose(curves, (1, 0, 2)).tolist()
    flat_ctrlpts = [pt for row in ctrlpts_grid for pt in row]  # flatten

    # Create and configure NURBS surface
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.set_ctrlpts(flat_ctrlpts, n_points, n_curves)

    # Auto-generate uniform knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

    # Optional visualization
    surf.vis = VisMPL.VisSurface()
    surf.render()

    return surf

def get_pca_plane_normal(points, mode='orthogonal_to_pc1'):
    pca = PCA(n_components=3)
    pca.fit(points)

    pc1, pc2, pc3 = pca.components_

    if mode == 'orthogonal_to_pc1':
        return pc1  # e.g. slice orthogonal to length
    elif mode == 'orthogonal_to_pc2':
        return pc2  # e.g. slice orthogonal to chord
    else:
        return pc3  # often close to surface normal

def find_best_cross_section_plane(points, normals, origin, input_normal, angle_step_deg=10, r=0.02):
    """
    Rotates planes around the input normal at the given origin and finds the best plane by energy.
    """
    def evaluate_plane_energy(points, normals, plane_point, plane_normal, r=0.02):
        """
        Computes the energy of a candidate plane according to Dimitrov et al.
        points: Nx3 point cloud
        normals: Nx3 array of normals
        plane_point: 3D point on the candidate plane
        plane_normal: normal of the candidate plane
        r: radius of influence
        """
        d = np.dot(points - plane_point, plane_normal)  # signed distances
        weight = np.exp(-3 * (d / r) ** 2)
        alignment = np.abs(np.dot(normals, plane_normal))
        energy = np.sum((1 - alignment) * weight)
        return energy
    
    best_energy = np.inf
    best_normal = None

    # Use PCA-provided input_normal as the base
    default_axis = np.array([0, 1, 0])
    if np.allclose(input_normal, default_axis):
        default_axis = np.array([0, 1, 0])
    rot_axis = np.cross(input_normal, default_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    for angle_deg in range(0, 180, angle_step_deg):
        rot = R.from_rotvec(np.deg2rad(angle_deg) * rot_axis)
        candidate_normal = rot.apply(input_normal)
        energy = evaluate_plane_energy(points, normals, origin, candidate_normal, r)
        if energy < best_energy:
            best_energy = energy
            best_normal = candidate_normal

    return origin, best_normal, best_energy


def visualize_surface_with_3d_spline(plotter, surf_poly, plane_origin, plane_normal, spline_3d):
    bounds = surf_poly.bounds
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 0.2
    plane = pv.Plane(center=plane_origin, direction=plane_normal, i_size=size, j_size=size)

    plotter.add_mesh(plane, color="lightblue", opacity=0.4)
    plotter.add_mesh(pv.PolyData(plane_origin), color='red', point_size=20, render_points_as_spheres=True)
    
    if spline_3d is not None and len(spline_3d) > 0:
        curve = pv.Spline(spline_3d, len(spline_3d) * 2)
        plotter.add_mesh(curve, color='blue', line_width=3, label='3D Spline')
    else:
        print("Spline is None")

def trace_mst_longest_path(points_2d):
    """
    Build a Minimum Spanning Tree (MST) from 2D points and extract the longest simple path.
    Returns the ordered path as an Nx2 array.
    """
    points = np.asarray(points_2d)
    n = len(points)

    # Step 1: Build complete distance matrix
    dists = distance_matrix(points, points)

    # Step 2: Create weighted graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dists[i, j])

    # Step 3: Build MST
    mst = nx.minimum_spanning_tree(G)

    # Step 4: Find endpoints (furthest apart in MST)
    longest_path = []
    max_length = 0

    for start in range(n):
        lengths, paths = nx.single_source_dijkstra(mst, start)
        far_node = max(lengths, key=lengths.get)
        if lengths[far_node] > max_length:
            max_length = lengths[far_node]
            longest_path = paths[far_node]

    return points[longest_path]

def fit_g2_connection(p_end, d_end, p_target, alpha=0.5):
    d = np.linalg.norm(p_target - p_end)
    handle_len = alpha * d
    control1 = p_end + d_end * handle_len
    control2 = p_target - d_end * handle_len
    return np.array([p_end, control1, control2, p_target])

def extend_spline_to_endpoints(path_2d, edge_le_2d, edge_te_2d, margin=1e-3):
    path = np.asarray(path_2d)
    ext = path.copy()

    dist_le = np.linalg.norm(path[0] - edge_le_2d)
    if dist_le > margin:
        d_start = path[1] - path[0]
        d_start /= np.linalg.norm(d_start) + 1e-8
        seg_le = fit_g2_connection(path[0], d_start, edge_le_2d)
        ext = np.vstack([seg_le[1:], ext])

    dist_te = np.linalg.norm(path[-1] - edge_te_2d)
    if dist_te > margin:
        d_end = path[-1] - path[-2]
        d_end /= np.linalg.norm(d_end) + 1e-8
        seg_te = fit_g2_connection(path[-1], d_end, edge_te_2d)
        ext = np.vstack([ext, seg_te[1:]])

    return ext

def update_le_te_from_curve(path_2d, uv_all, projected_all):
    """
    Given the traced path in 2D (path_2d) and all projected mesh points in UV and 3D space,
    find the closest mesh-projected points to the curve start and end.
    """
    start_uv = path_2d[0]
    end_uv   = path_2d[-1]

    dist_to_start = np.linalg.norm(uv_all - start_uv, axis=1)
    dist_to_end   = np.linalg.norm(uv_all - end_uv, axis=1)

    le_idx = np.argmin(dist_to_start)
    te_idx = np.argmin(dist_to_end)

    edge_le_2d = uv_all[le_idx]
    edge_te_2d = uv_all[te_idx]
    edge_le_3d = projected_all[le_idx]
    edge_te_3d = projected_all[te_idx]

    return edge_le_2d, edge_te_2d, edge_le_3d, edge_te_3d

def slice_and_fit_bspline(points, plane_origin, plane_normal, thickness=0.01, smoothing=0.001, num_samples=100):
    """
    Robust slicing, tracing, and spline extension using MST and edge-aware projection.
    """
    # 1. Slice
    d = np.dot(points - plane_origin, plane_normal)
    sliced_points = points[np.abs(d) <= thickness]
    plane_origin = np.mean(sliced_points, axis=0)
    if len(sliced_points) < 5:
        return sliced_points, None, None, None, None, None

    # 2. Project sliced points to slicing plane
    d_sliced = np.dot(sliced_points - plane_origin, plane_normal)
    projections = sliced_points - np.outer(d_sliced, plane_normal)
    plotter.add_points(pv.PolyData(projections), color='red', point_size=5)

    # 3. Create local frame
    u = np.cross(plane_normal, [1, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(plane_normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    u = np.array([1, 0, 0])
    v = np.array([0, 0, 1])

    # 4. Project to (u,v)
    local = projections - plane_origin
    x = np.dot(local, u)
    y = np.dot(local, v)
    projected_2d = np.column_stack((x, y))

    # 5. Trace ordered curve in 2D
    path_2d = trace_mst_longest_path(projected_2d)
    if len(path_2d) < 4:
        print("Could not trace a valid spline path — skipping slice.")
        return sliced_points, projected_2d, None, None, u, v
    
    # 6. Project all mesh points and find LE/TE in slicing band
    d_all = np.dot(points - plane_origin, plane_normal)
    mask = np.abs(d_all) <= thickness
    near_plane_points = points[mask]

    d_proj = np.dot(near_plane_points - plane_origin, plane_normal)
    projected_all = near_plane_points - np.outer(d_proj, plane_normal)
    local_all = projected_all - plane_origin
    u_vals_all = np.dot(local_all, u)
    v_vals_all = np.dot(local_all, v)
    uv_all = np.column_stack((u_vals_all, v_vals_all))

    edge_le_2d, edge_te_2d, edge_le_3d, edge_te_3d = update_le_te_from_curve(path_2d, uv_all, projected_all)

    dist_start_to_le = np.linalg.norm(path_2d[0] - edge_le_2d)
    dist_end_to_le = np.linalg.norm(path_2d[-1] - edge_le_2d)
    
    if dist_end_to_le < dist_start_to_le:
        print("Flipping path direction for consistency.")
        path_2d = path_2d[::-1]
    
    print("Start point vs LE:", path_2d[0], "->", edge_le_2d)
    print("End point vs TE:", path_2d[-1], "->", edge_te_2d)
    
    # 7. Extend path to physical LE/TE
    extended_path_2d = extend_spline_to_endpoints(path_2d, edge_le_2d, edge_te_2d)
    
    plotter.add_points(np.vstack([edge_le_3d, edge_te_3d]), color='green', point_size=15)
    
    # 8. Fit spline to extended path
    try:
        #tck, _ = splprep(extended_path_2d.T, s=smoothing)
        tck, _ = splprep(path_2d.T, s=smoothing)
        u_fine = np.linspace(0, 1, num_samples)
        x_spline, y_spline = splev(u_fine, tck)
        spline_2d = np.column_stack([x_spline, y_spline])
        spline_3d = plane_origin + spline_2d[:, 0:1] * u + spline_2d[:, 1:2] * v
    except Exception as e:
        print("Spline fitting failed:", e)
        return sliced_points, projected_2d, None, None, u, v

    return sliced_points, projected_2d, spline_2d, spline_3d, u, v




# Load your VTK mesh
filepath = os.path.join(os.getcwd(), "Inputs", "Mesh Data", "crm2.vtm")
mesh = load_mesh(filepath)
surf_names = mesh.get_surface_names()

idx = 8
surf = mesh.get_surface_mesh(surf_names[idx])
surf_poly = surf.extract_surface()
surf_with_normals = surf_poly.compute_normals(point_normals=True, auto_orient_normals=True)
points = surf_with_normals.points
normals = surf_with_normals.point_data['Normals']
#plane_normal = get_pca_plane_normal(points, mode='orthogonal_to_pc1')
plane_normal = np.array([0, 1, 0])

n_splines = 10
step_vector = np.array([0, 1, 0])
min_y = np.min(points[:, 1])
step_size = (np.max(points[:, 1]) - np.min(points[:, 1]))/n_splines 

plotter = pv.Plotter()
spline_list = []
for i in range(n_splines+1):
    yi = min_y + i * step_size
    print(f"Slice at Y = {yi:.3f}")
    
    candidates = points[np.isclose(points[:, 1], yi, atol=1e-1)]
    if len(candidates) == 0:
        print(f"No points found near Y = {yi:.3f}")
        continue
    
    plane_origin = candidates[np.argmin(candidates[:,0])]
    
    #plane_origin, plane_normal, energy = find_best_cross_section_plane(points, normals, plane_origin, plane_normal)
    sliced_pts, proj_2d, spline_2d, spline_3d, u, v = slice_and_fit_bspline(points, plane_origin, plane_normal, thickness=0.03, smoothing=0.001, num_samples=100 )

    if spline_3d is not None:
        visualize_surface_with_3d_spline(plotter, surf_poly, plane_origin, plane_normal, spline_3d)
        if spline_3d[0, 0] > spline_3d[-1, 0]:
            spline_3d = spline_3d[::-1]
        spline_list.append(spline_3d)

plotter.add_mesh(surf_poly, color="lightgray", opacity=0.3, show_edges=True)
plotter.add_axes()
plotter.show_bounds()
plotter.show_grid()
plotter.show()

print("spline_list length:", len(spline_list))
for i, s in enumerate(spline_list):
    print(f"Spline {i} shape:", s.shape)

surf = build_nurbs_surface_from_splines(spline_list, degree_u=3, degree_v=3)
step_path = os.path.join(os.getcwd(), "Outputs", "NURBs", f"surface_{idx}.step")
export_nurbs_to_step(surf, step_path)
