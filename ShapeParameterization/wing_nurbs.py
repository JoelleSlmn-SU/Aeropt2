import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree, distance_matrix, ConvexHull
from scipy.interpolate import splprep, splev, griddata
from sklearn.decomposition import PCA
from geomdl import BSpline, utilities, NURBS, fitting
from geomdl.visualization import VisMPL
import networkx as nx

class WingNURBSGenerator:
    def __init__(self, points, normals=None, mesh_connectivity=None,
                 n_span: int = 15, n_chord: int = 20):
        # keep both names to satisfy all methods that reference either
        pts = np.asarray(points, float)
        self.points = pts
        self.pts = pts

        self.normals = normals
        self.mesh_connectivity = mesh_connectivity
        self.wing_frame = None
        self.edges = {}

        # sampling/grid sizes used by generate()
        self.n_span = int(n_span)
        self.n_chord = int(n_chord)
        
    def detect_wing_orientation(self):
        """
        Use PCA to establish wing coordinate system.
        For typical wings: X=chordwise, Y=spanwise, Z=thickness
        """
        pca = PCA(n_components=3)
        pca.fit(self.points)
        
        # Wing coordinate system
        components = pca.components_
        
        # Identify spanwise direction (usually longest extent)
        extents = np.ptp(self.points @ components.T, axis=0)
        span_idx = np.argmax(extents)
        
        # Find chord direction (second longest extent)
        remaining_indices = [i for i in range(3) if i != span_idx]
        remaining_extents = extents[remaining_indices]
        chord_local_idx = np.argmax(remaining_extents)
        chord_idx = remaining_indices[chord_local_idx]
        
        # Thickness direction is the remaining one
        thick_idx = remaining_indices[1 - chord_local_idx]
        
        self.wing_frame = {
            'span': components[span_idx],    # Y direction
            'chord': components[chord_idx],  # X direction  
            'thickness': components[thick_idx], # Z direction
            'origin': np.mean(self.points, axis=0),
            'extents': extents,
            'span_range': None,
            'chord_range': None,
            'thickness_range': None
        }
        
        # Calculate coordinate ranges
        origin = self.wing_frame['origin']
        centered = self.points - origin
        self.wing_frame['chord_range'] = [
            np.min(np.dot(centered, self.wing_frame['chord'])),
            np.max(np.dot(centered, self.wing_frame['chord']))
        ]
        self.wing_frame['span_range'] = [
            np.min(np.dot(centered, self.wing_frame['span'])),
            np.max(np.dot(centered, self.wing_frame['span']))
        ]
        self.wing_frame['thickness_range'] = [
            np.min(np.dot(centered, self.wing_frame['thickness'])),
            np.max(np.dot(centered, self.wing_frame['thickness']))
        ]
        
        return self.wing_frame
    
    def detect_wing_edges(self):
        """
        Detect leading edge, trailing edge, wing tip, and wing root.
        Uses aerospace knowledge about wing geometry.
        """
        if self.wing_frame is None:
            self.detect_wing_orientation()
            
        # Transform points to wing coordinate system
        origin = self.wing_frame['origin']
        chord_dir = self.wing_frame['chord']
        span_dir = self.wing_frame['span']
        
        # Project points onto wing coordinate system
        centered = self.points - origin
        chord_coords = np.dot(centered, chord_dir)
        span_coords = np.dot(centered, span_dir)
        
        # Detect edges based on extrema
        chord_min_idx = np.argmin(chord_coords)
        chord_max_idx = np.argmax(chord_coords)
        span_min_idx = np.argmin(span_coords)
        span_max_idx = np.argmax(span_coords)
        
        self.edges = {
            'leading_edge': self.points[chord_min_idx],
            'trailing_edge': self.points[chord_max_idx], 
            'wing_root': self.points[span_min_idx],
            'wing_tip': self.points[span_max_idx]
        }
        
        le = self.edges['leading_edge']
        te = self.edges['trailing_edge']
        root = self.edges['wing_root']
        tip  = self.edges['wing_tip']

        ch = self.wing_frame['chord']   # or self.frame['chord']
        sp = self.wing_frame['span']    # or self.frame['span']

        if np.dot(ch, (te - le)) < 0.0:
            self.wing_frame['chord'] *= -1.0
        if np.dot(sp, (tip - root)) < 0.0:
            self.wing_frame['span']  *= -1.0
        self.wing_frame['normal'] = np.cross(self.wing_frame['chord'], self.wing_frame['span'])
        
        return self.edges
    
    def create_structured_grid_robust(self, n_chord=20, n_span=15):
        """
        Create robust structured grid using local averaging and smoothing.
        Much more stable than RBF for noisy data.
        """
        if self.wing_frame is None:
            self.detect_wing_orientation()
            
        origin = self.wing_frame['origin']
        chord_dir = self.wing_frame['chord']
        span_dir = self.wing_frame['span']
        thickness_dir = self.wing_frame['thickness']
        
        # Get coordinate ranges
        chord_range = self.wing_frame['chord_range']
        span_range = self.wing_frame['span_range']
        
        # Create regular grid in parameter space
        chord_vals = np.linspace(chord_range[0], chord_range[1], n_chord)
        span_vals = np.linspace(span_range[0], span_range[1], n_span)
        
        # Transform points to wing coordinates for efficient lookup
        centered = self.points - origin
        chord_coords = np.dot(centered, chord_dir)
        span_coords = np.dot(centered, span_dir)
        thickness_coords = np.dot(centered, thickness_dir)
        
        # Create grid using local neighborhood averaging
        grid_points = np.zeros((n_span, n_chord, 3))
        
        # Adaptive search radius based on point density
        avg_spacing = np.sqrt(np.ptp(chord_coords) * np.ptp(span_coords) / len(self.points))
        search_radius = avg_spacing * 3.0  # Adjust multiplier as needed
        
        tree = cKDTree(np.column_stack([chord_coords, span_coords]))
        
        for j, span_val in enumerate(span_vals):
            for i, chord_val in enumerate(chord_vals):
                query_point = np.array([chord_val, span_val])
                
                # Find nearby points in parameter space
                indices = tree.query_ball_point(query_point, search_radius)
                
                if len(indices) == 0:
                    # No points found, use nearest neighbor
                    _, idx = tree.query(query_point)
                    grid_points[j, i] = self.points[idx]
                else:
                    # Use distance-weighted average of nearby points
                    nearby_points_2d = np.column_stack([chord_coords[indices], 
                                                      span_coords[indices]])
                    distances = np.linalg.norm(nearby_points_2d - query_point, axis=1)
                    
                    # Avoid division by zero
                    distances = np.maximum(distances, 1e-10)
                    weights = 1.0 / distances
                    weights /= np.sum(weights)
                    
                    # Weighted average of 3D points
                    averaged_point = np.sum(self.points[indices] * weights[:, np.newaxis], axis=0)
                    grid_points[j, i] = averaged_point
            chord_dir = self.wing_frame['chord']
            proj = grid_points[j].dot(chord_dir)
            order = np.argsort(proj)
            grid_points[j] = grid_points[j, order]
        
        span_dir = self.wing_frame['span']
        span_proj = np.array([grid_points[j].mean(axis=0).dot(span_dir) for j in range(n_span)])
        span_order = np.argsort(span_proj)
        grid_points = grid_points[span_order]
        
        return grid_points
    
    def create_structured_grid_rbf_robust(self, n_chord=20, n_span=15):
        """
        Robust RBF interpolation with outlier removal and regularization.
        """
        if self.wing_frame is None:
            self.detect_wing_orientation()
            
        origin = self.wing_frame['origin']
        chord_dir = self.wing_frame['chord']
        span_dir = self.wing_frame['span']
        thickness_dir = self.wing_frame['thickness']
        
        # Transform points to wing coordinates
        centered = self.points - origin
        chord_coords = np.dot(centered, chord_dir)
        span_coords = np.dot(centered, span_dir)
        thickness_coords = np.dot(centered, thickness_dir)
        
        # Remove outliers in thickness direction
        thickness_mean = np.mean(thickness_coords)
        thickness_std = np.std(thickness_coords)
        thickness_threshold = 3.0 * thickness_std
        
        valid_mask = np.abs(thickness_coords - thickness_mean) < thickness_threshold
        chord_coords_clean = chord_coords[valid_mask]
        span_coords_clean = span_coords[valid_mask]
        thickness_coords_clean = thickness_coords[valid_mask]
        
        print(f"Removed {np.sum(~valid_mask)} outliers out of {len(self.points)} points")
        
        # Create regular grid in parameter space
        chord_range = [chord_coords_clean.min(), chord_coords_clean.max()]
        span_range = [span_coords_clean.min(), span_coords_clean.max()]
        
        chord_vals = np.linspace(chord_range[0], chord_range[1], n_chord)
        span_vals = np.linspace(span_range[0], span_range[1], n_span)
        
        chord_grid, span_grid = np.meshgrid(chord_vals, span_vals)
        query_points = np.column_stack([chord_grid.ravel(), span_grid.ravel()])
        
        # Use griddata with cubic interpolation (more stable than RBF)
        try:
            thickness_interp = griddata(
                (chord_coords_clean, span_coords_clean), 
                thickness_coords_clean,
                query_points, 
                method='cubic', 
                fill_value=0.0
            )
            
            # Fill any remaining NaN values with linear interpolation
            nan_mask = np.isnan(thickness_interp)
            if np.any(nan_mask):
                thickness_interp[nan_mask] = griddata(
                    (chord_coords_clean, span_coords_clean), 
                    thickness_coords_clean,
                    query_points[nan_mask], 
                    method='linear', 
                    fill_value=0.0
                )
                
        except Exception as e:
            print(f"Cubic interpolation failed: {e}, using linear")
            thickness_interp = griddata(
                (chord_coords_clean, span_coords_clean), 
                thickness_coords_clean,
                query_points, 
                method='linear', 
                fill_value=0.0
            )
        
        # Reconstruct 3D points
        grid_3d = []
        for i, (c, s, t) in enumerate(zip(query_points[:, 0], 
                                        query_points[:, 1], 
                                        thickness_interp)):
            if np.isnan(t):
                t = 0.0  # Fallback for any remaining NaN values
            point_3d = origin + c * chord_dir + s * span_dir + t * thickness_dir
            grid_3d.append(point_3d)
            
        grid_3d = np.array(grid_3d).reshape(n_span, n_chord, 3)
        return grid_3d
    
    def create_structured_grid_legacy(self, n_chord=20, n_span=15):
        """
        Legacy structured grid method using nearest neighbors.
        Kept for backward compatibility.
        """
        if self.wing_frame is None:
            self.detect_wing_orientation()
            
        origin = self.wing_frame['origin']
        chord_dir = self.wing_frame['chord']
        span_dir = self.wing_frame['span']
        
        # Transform points to wing coordinates
        centered = self.points - origin
        chord_coords = np.dot(centered, chord_dir)
        span_coords = np.dot(centered, span_dir)
        
        # Create structured parameter space
        chord_range = np.linspace(chord_coords.min(), chord_coords.max(), n_chord)
        span_range = np.linspace(span_coords.min(), span_coords.max(), n_span)
        
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(self.points)
        
        grid_points = []
        for j, span_val in enumerate(span_range):
            chord_line = []
            for i, chord_val in enumerate(chord_range):
                # Reconstruct 3D position
                grid_pos_3d = origin + chord_val * chord_dir + span_val * span_dir
                
                # Find closest mesh point
                dist, idx = tree.query(grid_pos_3d)
                closest_point = self.points[idx]
                
                # If too far, interpolate from nearby points
                if dist > 0.1:  # threshold
                    dists, indices = tree.query(grid_pos_3d, k=5)
                    weights = 1 / (dists + 1e-8)
                    weights /= np.sum(weights)
                    closest_point = np.sum(self.points[indices] * weights[:, None], axis=0)
                
                chord_line.append(closest_point)
            grid_points.append(chord_line)
            
        return np.array(grid_points)
    
    def extract_airfoil_sections_improved(self, n_sections=10):
        """
        Extract airfoil sections using proper slicing and ordering.
        """
        if self.wing_frame is None:
            self.detect_wing_orientation()
            
        origin = self.wing_frame['origin']
        span_dir = self.wing_frame['span']
        chord_dir = self.wing_frame['chord']
        thickness_dir = self.wing_frame['thickness']
        
        # Get span range
        span_range = self.wing_frame['span_range']
        span_positions = np.linspace(span_range[0], span_range[1], n_sections)
        
        airfoil_curves = []
        
        for span_pos in span_positions:
            # Extract points near this span station
            section_center = origin + span_pos * span_dir
            distances_to_plane = np.abs(np.dot(self.points - section_center, span_dir))
            
            # Adaptive thickness based on wing size
            wing_span = span_range[1] - span_range[0]
            section_thickness = wing_span * 0.05  # 5% of span
            section_mask = distances_to_plane < section_thickness
            section_points = self.points[section_mask]
            
            if len(section_points) < 8:
                continue
                
            # Project to section plane (chord-thickness plane)
            local_points = section_points - section_center
            chord_coords = np.dot(local_points, chord_dir)
            thickness_coords = np.dot(local_points, thickness_dir)
            
            # Create ordered airfoil curve
            points_2d = np.column_stack([chord_coords, thickness_coords])
            
            try:
                # Sort points to create proper airfoil ordering
                # Start from leading edge, go around the airfoil
                center_2d = np.mean(points_2d, axis=0)
                angles = np.arctan2(points_2d[:, 1] - center_2d[1], 
                                  points_2d[:, 0] - center_2d[0])
                
                # Sort by angle to create ordered curve
                sort_indices = np.argsort(angles)
                ordered_section = section_points[sort_indices]
                
                # Smooth with spline if enough points
                if len(ordered_section) > 10:
                    # Close the curve
                    closed_section = np.vstack([ordered_section, ordered_section[0]])
                    tck, _ = splprep(closed_section.T, s=0.01, per=True)
                    u = np.linspace(0, 1, 50)
                    smooth_curve = np.array(splev(u, tck)).T
                    airfoil_curves.append(smooth_curve)
                else:
                    airfoil_curves.append(ordered_section)
                    
            except Exception as e:
                print(f"Failed to process airfoil section at span {span_pos}: {e}")
                continue
                
        return airfoil_curves
    
    def _compute_uv(self):
        """Project points to a stable 2D param plane (span/chord) with sign-fixed axes."""
        # detect/load frame once
        if getattr(self, "wing_frame", None) is None:
            o, c, s = self.pts.mean(axis=0), None, None
            # PCA axes
            X = self.pts - o
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            c = Vt[0]  # chord-ish
            s = Vt[1]  # span-ish
            # lock orientation with extrema
            pc = (self.pts - o) @ c
            ps = (self.pts - o) @ s
            le = self.pts[np.argmin(pc)]; te = self.pts[np.argmax(pc)]
            root = self.pts[np.argmin(ps)]; tip = self.pts[np.argmax(ps)]
            if np.dot(c, te - le) < 0: c = -c
            if np.dot(s, tip - root) < 0: s = -s
            self.wing_frame = {"origin": o, "chord": c/np.linalg.norm(c), "span": s/np.linalg.norm(s)}

        o = self.wing_frame["origin"]; c = self.wing_frame["chord"]; s = self.wing_frame["span"]
        # 2D coordinates in (span, chord) param plane
        SC = np.c_[ (self.pts - o) @ s, (self.pts - o) @ c ]  # [N, 2]  (v,u) order
        # normalize to [0,1]^2 just for a stable sampling box
        mn = SC.min(axis=0); mx = SC.max(axis=0); rng = np.maximum(mx - mn, 1e-12)
        uv = (SC - mn) / rng
        return uv, (mn, mx, rng), self.wing_frame

    def _sample_lattice_from_uv(self, uv, k=12, nu_samp=None, nv_samp=None):
        """
        Build a regular (v,u) lattice of 3D samples via inverse-distance weights
        in UV plane. If k=1, this reduces to nearest-neighbour like you suggested.
        Returns: lattice_points with shape (nv_samp, nu_samp, 3)
        """
        nu = int(nu_samp or max(6, self.n_chord))
        nv = int(nv_samp or max(6, self.n_span))
        us = np.linspace(0.0, 1.0, nu)
        vs = np.linspace(0.0, 1.0, nv)

        kdt = cKDTree(uv)
        lattice = np.zeros((nv, nu, 3), dtype=float)

        for j, v in enumerate(vs):
            for i, u in enumerate(us):
                d, idx = kdt.query([v, u], k=min(k, len(self.pts)))
                if np.isscalar(d):  # k==1 → scalar
                    lattice[j, i] = self.pts[idx]
                else:
                    w = 1.0 / np.maximum(d, 1e-12)
                    w /= w.sum()
                    lattice[j, i] = (w[:, None] * self.pts[idx]).sum(axis=0)
        return lattice  # (nv, nu, 3)
    
    def visualize_analysis(self):
        """Visualize the wing analysis results."""
        plotter = pv.Plotter(shape=(2, 2))
        
        # Original points
        plotter.subplot(0, 0)
        plotter.add_points(self.points, color='blue', point_size=2)
        plotter.add_title("Original Point Cloud")
        
        # Wing frame
        if self.wing_frame is not None:
            plotter.subplot(0, 1)
            plotter.add_points(self.points, color='lightgray', point_size=2)
            
            origin = self.wing_frame['origin']
            scale = np.ptp(self.points, axis=0).max() * 0.3
            
            # Add coordinate axes
            chord_end = origin + scale * self.wing_frame['chord']
            span_end = origin + scale * self.wing_frame['span'] 
            thick_end = origin + scale * self.wing_frame['thickness']
            
            plotter.add_lines(np.array([origin, chord_end]), color='red', width=5)
            plotter.add_lines(np.array([origin, span_end]), color='green', width=5) 
            plotter.add_lines(np.array([origin, thick_end]), color='blue', width=5)
            plotter.add_title("Wing Coordinate System")
        
        # Detected edges
        if self.edges:
            plotter.subplot(1, 0)
            plotter.add_points(self.points, color='lightgray', point_size=2)
            
            edge_points = np.array(list(self.edges.values()))
            colors = ['red', 'blue', 'green', 'orange']
            labels = list(self.edges.keys())
            
            for i, (point, color, label) in enumerate(zip(edge_points, colors, labels)):
                plotter.add_points(point, color=color, point_size=10, 
                                 render_points_as_spheres=True)
                
            plotter.add_title("Detected Wing Features")
        
        # Robust grid preview
        plotter.subplot(1, 1)
        try:
            grid = self.create_structured_grid_robust(15, 10)
            grid_mesh = pv.StructuredGrid()
            grid_mesh.points = grid.reshape(-1, 3)
            grid_mesh.dimensions = grid.shape[:2][::-1] + (1,)
            plotter.add_mesh(grid_mesh, show_edges=True, line_width=2, opacity=0.7)
            plotter.add_title("Robust Interpolated Grid")
        except Exception as e:
            plotter.add_text(f"Grid generation failed: {str(e)}", font_size=10)
            
        plotter.show()

    def build_surface(self, grid_3d):
        # grid_3d shape: (nv, nu, 3)
        nv, nu, _ = grid_3d.shape
        pts_flat = [p.tolist() for row in grid_3d for p in row]
        deg_u, deg_v = 3, 3
        # choose a reasonable control net density relative to samples
        ctrl_u = max(4, nu // 1)
        ctrl_v = max(4, nv // 1)
        try:
            surf = fitting.approximate_surface(
                pts_flat,
                size_u=nu, size_v=nv,
                degree_u=deg_u, degree_v=deg_v,
                ctrlpts_size_u=ctrl_u, ctrlpts_size_v=ctrl_v
            )
        except TypeError:
            # geomdl without ctrlpts_size_* kwargs
            surf = fitting.approximate_surface(
                pts_flat,
                size_u=nu, size_v=nv,
                degree_u=deg_u, degree_v=deg_v
            )
        try:
            surf.delta = 0.03
        except Exception:
            pass
        return surf

    # Provide the method that process_wing_surface() expects
    def generate_nurbs_surface(self, method='robust_grid'):
        m = (method or 'robust_grid').lower()
        if m in ('robust_grid', 'structured_grid'):
            grid = self.create_structured_grid_robust(
                n_chord=self.n_chord, n_span=self.n_span
            )
            return self.build_surface(grid)
        elif m in ('rbf_grid', 'rbf'):
            grid = self.create_structured_grid_rbf_robust(
                n_chord=self.n_chord, n_span=self.n_span
            )
            return self.build_surface(grid)
        else:
            # fallback to the least-squares path already implemented
            return self.generate()


# Updated process function
def process_wing_surface(points, normals=None, method='robust_grid'):
    generator = WingNURBSGenerator(points, normals)
    generator.detect_wing_orientation()
    generator.detect_wing_edges()
    surface = generator.generate_nurbs_surface(method=method)
    return surface, generator