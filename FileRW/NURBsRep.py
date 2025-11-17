import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

def visual_nurbs_from_mesh(mesh: pv.PolyData,
                           grid_res=(60, 60),     # smooth surface resolution
                           ctrl_res=(6, 6),       # control net resolution
                           smooth_method='cubic', # 'linear' if cubic fails
                           ctrl_method='linear'):
    """
    Build a *visual* NURBS-like surface + control net from an arbitrary triangulated mesh.
    Returns: surface_actor, control_net_actor, control_pts_actor
    """
    # 1) Get point cloud
    P = mesh.points.astype(float)                 # (N,3)
    Pmean = P.mean(0)
    X = P - Pmean

    # 2) PCA basis (best-fit plane axes e0,e1 and normal e2)
    U, S, Vt = np.linalg.svd(np.cov(X.T))
    e0, e1, e2 = Vt[0], Vt[1], Vt[2]

    # 3) Project to 2D plane → (u,v)
    uv = np.c_[X @ e0, X @ e1]
    umin, vmin = uv.min(0)
    umax, vmax = uv.max(0)

    # 4) Build regular (u,v) grid
    nu, nv = grid_res
    ug = np.linspace(umin, umax, nu)
    vg = np.linspace(vmin, vmax, nv)
    UG, VG = np.meshgrid(ug, vg)                  # shape (nv, nu)

    # 5) Interpolate original surface onto the grid (component-wise)
    #    We reconstruct XYZ by fitting x(u,v), y(u,v), z(u,v)
    Xg = griddata(uv, P[:,0], (UG, VG), method=smooth_method)
    Yg = griddata(uv, P[:,1], (UG, VG), method=smooth_method)
    Zg = griddata(uv, P[:,2], (UG, VG), method=smooth_method)

    # Fallback if cubic fails (NaNs at edges)
    if np.isnan(Xg).any() or np.isnan(Yg).any() or np.isnan(Zg).any():
        Xg = griddata(uv, P[:,0], (UG, VG), method='linear')
        Yg = griddata(uv, P[:,1], (UG, VG), method='linear')
        Zg = griddata(uv, P[:,2], (UG, VG), method='linear')

    # 6) Build a StructuredGrid for the smooth surface
    surf = pv.StructuredGrid()
    surf.points = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]
    surf.dimensions = [nu, nv, 1]

    # 7) Choose "control" parameter lines and sample 3D positions there
    cu, cv = ctrl_res
    u_ctrl = np.linspace(umin, umax, cu)
    v_ctrl = np.linspace(vmin, vmax, cv)
    Uc, Vc = np.meshgrid(u_ctrl, v_ctrl)          # (cv, cu)

    Xc = griddata(uv, P[:,0], (Uc, Vc), method=ctrl_method)
    Yc = griddata(uv, P[:,1], (Uc, Vc), method=ctrl_method)
    Zc = griddata(uv, P[:,2], (Uc, Vc), method=ctrl_method)
    ctrl_pts = np.c_[Xc.ravel(), Yc.ravel(), Zc.ravel()]

    # 8) Build polylines for the control polygon (rows and columns)
    lines = []
    # rows (v fixed, u increasing)
    for j in range(cv):
        ids = np.arange(j*cu, (j+1)*cu, dtype=np.int32)
        lines.append(np.r_[len(ids), ids])
    # columns (u fixed, v increasing)
    for i in range(cu):
        ids = np.arange(i, cu*cv, cu, dtype=np.int32)
        lines.append(np.r_[len(ids), ids])
    lines = np.concatenate(lines)

    ctrl_poly = pv.PolyData(ctrl_pts)
    ctrl_poly.lines = lines

    # 9) Create actors (surface + net + points)
    surface_actor = surf.plot(texture=False, show_edges=False)  # if using standalone
    # For embedding in your PyVista Plotter:
    # return surf, ctrl_poly, pv.PolyData(ctrl_pts)

    return surf, ctrl_poly, pv.PolyData(ctrl_pts)

# ------- Example usage in a PyVista Plotter -------
# mesh = <your pv.PolyData surface>  # e.g., the pale-blue patch in your screenshot
# p = pv.Plotter()
# surf, ctrl_net, ctrl_pts = visual_nurbs_from_mesh(mesh,
#                                                   grid_res=(80,80),
#                                                   ctrl_res=(7,7))
# p.add_mesh(surf, color='white', smooth_shading=True, opacity=0.75)
# p.add_mesh(ctrl_net, color='black', line_width=2)
# p.add_mesh(ctrl_pts, color='black', point_size=14, render_points_as_spheres=True)
# p.show()

#import pyvista as pv
#from pathlib import Path

# Path to your VTM file
#vtm_path = Path(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Corner Bump Surface\corner.vtm")

# Load the multiblock dataset
#multi = pv.read(vtm_path)

#print(f"Number of blocks: {len(multi)}")
#print(multi.keys())   # to see block names / surface IDs

# Pick one surface (e.g. the one you want to visualise)
#mesh = multi[13]

#surf, ctrl_net, ctrl_pts = visual_nurbs_from_mesh(mesh, grid_res=(200, 200), ctrl_res=(20, 20))

#plotter = pv.Plotter()
#plotter.add_mesh(surf, color="lightblue", smooth_shading=True, opacity=0.7)
#plotter.add_mesh(ctrl_net, color="black", line_width=2)
#plotter.add_mesh(ctrl_pts, color="black", point_size=12, render_points_as_spheres=True)
#plotter.show()

from pathlib import Path
import matplotlib as plt

from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.Geom import Geom_BSplineSurface
from OCP.GeomConvert import GeomConvert


def _downcast_face(shape: TopoDS_Shape) -> TopoDS_Face:
    """Return a TopoDS_Face by DownCast or by exploring the first face inside."""
    f = TopoDS_Face.DownCast(shape)
    if (f is not None) and (not f.IsNull()):
        return f
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    if ex.More():
        f2 = TopoDS_Face.DownCast(ex.Current())
        if (f2 is not None) and (not f2.IsNull()):
            return f2
    raise RuntimeError("Shape is not (or does not contain) a Face")


def load_step_faces(step_path: str):
    r = STEPControl_Reader()
    if r.ReadFile(str(step_path)) != 1:
        raise RuntimeError("Failed to read STEP")
    r.TransferRoots()
    shape = r.OneShape()

    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS_Face.DownCast(ex.Current())
        if (f is not None) and (not f.IsNull()):
            faces.append(f)
        ex.Next()
    return faces


def face_to_bspline(face: TopoDS_Face):
    """
    Convert a (possibly analytic/trimmed) FACE to a BSpline surface.
    Returns: (bspline_surface, umin, umax, vmin, vmax)
    """
    # Convert THIS face to NURBS
    conv = BRepBuilderAPI_NurbsConvert(face, True)
    nurbs_face = _downcast_face(conv.Shape())

    # Underlying surface
    h_surf = BRep_Tool.Surface(nurbs_face)

    # Try direct downcast; if not BSpline, convert explicitly
    bs = Geom_BSplineSurface.DownCast(h_surf)
    if bs is None:
        bs = GeomConvert.ToBSplineSurface(h_surf)
    if bs is None:
        raise RuntimeError("Could not obtain a Geom_BSplineSurface")

    # Parametric bounds of this (possibly trimmed) face
    umin, umax, vmin, vmax = BRepTools.UVBounds(nurbs_face)
    return bs, umin, umax, vmin, vmax

    import numpy as np

def bspline_info(bs):
    Udeg, Vdeg = bs.UDegree(), bs.VDegree()
    nu, nv = bs.NbUPoles(), bs.NbVPoles()

    Uknots = [bs.UKnot(i+1) for i in range(bs.NbUKnots())]
    Vknots = [bs.VKnot(i+1) for i in range(bs.NbVKnots())]
    Umults = [bs.UMultiplicity(i+1) for i in range(bs.NbUKnots())]
    Vmults = [bs.VMultiplicity(i+1) for i in range(bs.NbVKnots())]

    poles = np.zeros((nv, nu, 3))
    weights = np.ones((nv, nu))
    for v in range(1, nv+1):
        for u in range(1, nu+1):
            P = bs.Pole(u, v)
            poles[v-1, u-1] = [P.X(), P.Y(), P.Z()]
            if bs.IsRational():
                weights[v-1, u-1] = bs.Weight(u, v)
    return Udeg, Vdeg, Uknots, Vknots, Umults, Vmults, poles, weights


def show_control_net(poles):
    """Quick matplotlib visual of the control polygon."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nv, nu, _ = poles.shape

    # draw rows (v fixed)
    for v in range(nv):
        ax.plot(poles[v,:,0], poles[v,:,1], poles[v,:,2], linewidth=1.5, color='k')
    # draw cols (u fixed)
    for u in range(nu):
        ax.plot(poles[:,u,0], poles[:,u,1], poles[:,u,2], linewidth=1.5, color='k')
    ax.scatter(poles[...,0], poles[...,1], poles[...,2], s=18, color='k')
    ax.set_box_aspect([1,1,0.6])
    plt.show()


if __name__ == "__main__":
    step_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Corner Bump CAD\Plattenmodell_CornerBump_2025-07-03_cf.stp"
    faces = load_step_faces(step_path)
    print(f"STEP has {len(faces)} faces")

    idx = 0  # pick the face you want
    bs, umin, umax, vmin, vmax = face_to_bspline(faces[idx])
    print("UDegree, VDegree:", bs.UDegree(), bs.VDegree())
    print("UV bounds:", (umin, umax, vmin, vmax))