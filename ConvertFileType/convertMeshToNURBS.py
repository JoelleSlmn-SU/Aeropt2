import os, sys
import numpy as np
import pyvista as pv
import rhinoinside
 
sys.path.append(os.path.dirname("FileRW"))
from FileRW.StpFile import step_to_stl
from convertFroToStl import convert_fro_to_stl
 
rhinoinside.load()
 
import Rhino
 
def split_stl_components(stl_path):
    import pyvista as pv
 
    mesh = pv.read(stl_path).extract_surface().triangulate()
    conn = mesh.connectivity()
 
    ids = sorted(set(conn["RegionId"]))
    components = []
 
    for rid in ids:
        comp = conn.threshold([rid - 0.5, rid + 0.5], scalars="RegionId")
        comp = comp.extract_surface().triangulate()
        components.append(comp)
        print(f"Component {rid}: points={comp.n_points}, faces={comp.n_cells}")
 
    return components
 
def save_components_as_stl(stl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
 
    comps = split_stl_components(stl_path)
    paths = []
 
    for i, comp in enumerate(comps):
        path = os.path.join(out_dir, f"component_{i}.stl")
        comp.save(path)
        paths.append(path)
        print(f"Saved component {i}: {path}")
 
    return paths
 
def component_mesh_to_nurbs_step(component_stl_path, step_path):
    component_stl_path = os.path.abspath(component_stl_path)
    step_path = os.path.abspath(step_path)
 
    doc = Rhino.RhinoDoc.CreateHeadless(None)
 
    mesh = pyvista_stl_to_rhino_mesh(component_stl_path)
 
    print("Component vertices:", mesh.Vertices.Count)
    print("Component faces:", mesh.Faces.Count)
 
    # This creates one NURBS/trimmed Brep face per mesh face.
    brep = Rhino.Geometry.Brep.CreateFromMesh(mesh, True)
    if brep is None:
        raise RuntimeError("CreateFromMesh failed.")
 
    doc.Objects.AddBrep(brep)
 
    options = Rhino.FileIO.FileWriteOptions()
    options.SuppressDialogBoxes = True
 
    ok = doc.WriteFile(step_path, options)
    if not ok:
        raise RuntimeError("Failed to write STEP.")
 
    print("Wrote component mesh-NURBS STEP:", step_path)
 
def pyvista_stl_to_rhino_mesh(stl_path):
    pvmesh = pv.read(stl_path).extract_surface().triangulate()
 
    points = np.asarray(pvmesh.points, float)
    faces = pvmesh.faces.reshape((-1, 4))[:, 1:4]
 
    mesh = Rhino.Geometry.Mesh()
 
    for p in points:
        mesh.Vertices.Add(float(p[0]), float(p[1]), float(p[2]))
 
    for f in faces:
        mesh.Faces.AddFace(int(f[0]), int(f[1]), int(f[2]))
 
    mesh.Normals.ComputeNormals()
    mesh.Compact()
 
    return mesh
 
def stl_to_step_with_rhino(stl_path, step_path):
    stl_path = os.path.abspath(stl_path)
    step_path = os.path.abspath(step_path)
 
    doc = Rhino.RhinoDoc.Create(None)
    Rhino.RhinoDoc.ActiveDoc = doc
 
    mesh = pyvista_stl_to_rhino_mesh(stl_path)
 
    print("Vertices:", mesh.Vertices.Count)
    print("Faces:", mesh.Faces.Count)
 
    brep = Rhino.Geometry.Brep.CreateFromMesh(mesh, True)
    if brep is None:
        raise RuntimeError("Brep.CreateFromMesh failed.")
 
    doc.Objects.AddBrep(brep)
 
    options = Rhino.FileIO.FileWriteOptions()
    options.SuppressDialogBoxes = True
 
    ok = doc.WriteFile(step_path, options)
    if not ok:
        raise RuntimeError("Rhino failed to write STEP file.")
 
    print("Wrote STEP:", step_path)
 
def stl_to_single_patch_step(stl_path, step_path, max_points=3000):
    import numpy as np
    import pyvista as pv
    import Rhino
    import System
 
    stl_path = os.path.abspath(stl_path)
    step_path = os.path.abspath(step_path)
 
    doc = Rhino.RhinoDoc.CreateHeadless(None)
 
    pvmesh = pv.read(stl_path).extract_surface().triangulate()
    pts = np.asarray(pvmesh.points, float)
 
    # downsample so Patch is not overloaded
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
 
    rh_pts = System.Collections.Generic.List[Rhino.Geometry.GeometryBase]()
 
    for p in pts:
        rh_pts.Add(Rhino.Geometry.Point(Rhino.Geometry.Point3d(
            float(p[0]), float(p[1]), float(p[2])
        )))
 
    print(f"Patch input points: {len(pts)}")
 
    brep = Rhino.Geometry.Brep.CreatePatch(
        rh_pts,
        20,      # u spans
        20,      # v spans
        1e-3     # tolerance
    )
 
    if brep is None:
        raise RuntimeError("Brep.CreatePatch failed.")
 
    doc.Objects.AddBrep(brep)
 
    options = Rhino.FileIO.FileWriteOptions()
    options.SuppressDialogBoxes = True
 
    ok = doc.WriteFile(step_path, options)
    if not ok:
        raise RuntimeError("Rhino failed to write patch STEP.")
 
    print("Wrote smooth patch STEP:", step_path)
    
 
 
def print_doc_objects(doc, tag):
    """Print all Rhino document objects after a command, so we can see where the pipeline fails."""
    objs = list(doc.Objects)
    print(f"\n[{tag}] object count = {len(objs)}")
    for i, obj in enumerate(objs):
        geo = obj.Geometry
        print(f"  {i:03d} | id={obj.Id} | type={obj.ObjectType} | geo={type(geo)}")
 
 
def save_debug_3dm(doc, out_dir, tag):
    """Save the intermediate Rhino document state for visual debugging."""
    os.makedirs(out_dir, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in tag)
    path = os.path.join(out_dir, f"debug_{safe}.3dm")
    opts = Rhino.FileIO.FileWriteOptions()
    opts.SuppressDialogBoxes = True
    ok = doc.WriteFile(path, opts)
    print(f"[DEBUG] saved {path} ok={ok}")
    return path
 
 
def run_rhino_command(doc, command, tag=None, debug_dir=None):
    """Run a Rhino command against the active Rhino document and report the result."""
    Rhino.RhinoDoc.ActiveDoc = doc
    tag = tag or command.split()[0].replace('_', '')
    print(f"\n[RHINO-CMD] {command}")
    ok = Rhino.RhinoApp.RunScript(command, True)
    print(f"[RHINO-CMD] ok={ok}")
    print_doc_objects(doc, tag)
    if debug_dir:
        save_debug_3dm(doc, debug_dir, tag)
    return ok
 
 
 
 
def run_first_working_command(doc, commands, tag, debug_dir=None):
    """
    Try several command-string variants. Rhino command options can be picky
    through RhinoInside, so this lets us test robustly without changing the main code.
    """
    before_count = len(list(doc.Objects))
    before_ids = {obj.Id for obj in doc.Objects}
 
    for cmd in commands:
        ok = run_rhino_command(doc, cmd, tag=tag, debug_dir=debug_dir)
        after_ids = {obj.Id for obj in doc.Objects}
        new_ids = after_ids - before_ids
        after_count = len(after_ids)
 
        if ok or new_ids or after_count != before_count:
            print(f"[DEBUG] accepted command for {tag}: {cmd}")
            return True
 
    print(f"[DEBUG][WARN] all command variants failed for {tag}")
    return False
 
def select_all_objects(doc):
    doc.Objects.UnselectAll()
    count = 0
    for obj in doc.Objects:
        if doc.Objects.Select(obj.Id):
            count += 1
    print(f"[DEBUG] selected {count} object(s)")
    return count
 
 
def select_new_objects(doc, before_ids):
    """Select objects created after a command. Falls back to all objects if none are new."""
    before_ids = set(before_ids)
    doc.Objects.UnselectAll()
    new_ids = []
    for obj in doc.Objects:
        if obj.Id not in before_ids:
            doc.Objects.Select(obj.Id)
            new_ids.append(obj.Id)
    if not new_ids:
        print("[DEBUG] no new objects detected; selecting all objects as fallback")
        select_all_objects(doc)
    else:
        print(f"[DEBUG] selected {len(new_ids)} new object(s)")
    return new_ids
 
 
def mesh_to_low_patch_nurbs_step(component_stl_path, out_step_path=None,
                                  target_quads=1000, detect_hard_edges=False,
                                  save_debug_stages=False):
    """
    STL mesh -> QuadRemesh -> SubD -> NURBS Brep -> STEP, driven entirely
    through in-memory RhinoCommon geometry calls.
 
    Deliberately does NOT use Rhino.RhinoApp.RunScript / command strings.
    Those route through the command-line interpreter, which expects a live
    Rhino application (message loop, active view, and in QuadRemesh's case
    a modal options dialog). RhinoInside headless docs don't provide that,
    so RunScript either no-ops or returns True without doing anything -
    which is exactly the silent failure you were seeing.
 
    QuadRemesh, SubD.CreateFromMesh and SubD.ToBrep are plain RhinoCommon
    methods (Rhino 7+) that operate directly on geometry objects with no
    document/view/dialog involved, so they work identically headless or not.
    """
    import Rhino
 
    component_stl_path = os.path.abspath(component_stl_path)
    if out_step_path is None:
        out_step_path = component_stl_path.replace('.stl', '_low_patch.stp')
    out_step_path = os.path.abspath(out_step_path)
 
    mesh = pyvista_stl_to_rhino_mesh(component_stl_path)
    print(f"Input mesh: vertices={mesh.Vertices.Count}, faces={mesh.Faces.Count}")
 
    # --- 1. Quad remesh -------------------------------------------------
    qr_params = Rhino.Geometry.QuadRemeshParameters()
    qr_params.TargetQuadCount = int(target_quads)
    qr_params.AdaptiveSize = True
    qr_params.DetectHardEdges = bool(detect_hard_edges)
 
    quad_mesh = mesh.QuadRemesh(qr_params)
    if quad_mesh is None:
        raise RuntimeError(
            "Mesh.QuadRemesh returned None. Check the input mesh is a single "
            "closed/manifold component - QuadRemesh is picky about non-manifold "
            "or multi-shell input."
        )
    print(f"Quad mesh: vertices={quad_mesh.Vertices.Count}, faces={quad_mesh.Faces.Count}")
 
    # --- 2. Quad mesh -> SubD -------------------------------------------
    subd = Rhino.Geometry.SubD.CreateFromMesh(quad_mesh, Rhino.Geometry.SubDCreationOptions())
    if subd is None:
        raise RuntimeError("SubD.CreateFromMesh returned None.")
    print(f"SubD: faces={subd.Faces.Count}, edges={subd.Edges.Count}, vertices={subd.Vertices.Count}")
 
    # --- 3. SubD -> NURBS Brep ------------------------------------------
    brep = subd.ToBrep(Rhino.Geometry.SubDToBrepOptions.Default)
    if brep is None:
        raise RuntimeError("SubD.ToBrep returned None.")
    print(f"NURBS Brep: faces={brep.Faces.Count}, surfaces={brep.Surfaces.Count}")
 
    # --- 4. Write STEP ----------------------------------------------------
    doc = Rhino.RhinoDoc.CreateHeadless(None)
    doc.Objects.AddBrep(brep)
 
    if save_debug_stages:
        debug_dir = os.path.join(os.path.dirname(out_step_path), 'debug_rhino')
        os.makedirs(debug_dir, exist_ok=True)
        debug_doc = Rhino.RhinoDoc.CreateHeadless(None)
        debug_doc.Objects.AddMesh(quad_mesh)
        debug_doc.Objects.AddSubD(subd)
        debug_doc.Objects.AddBrep(brep)
        opts = Rhino.FileIO.FileWriteOptions()
        opts.SuppressDialogBoxes = True
        debug_doc.WriteFile(os.path.join(debug_dir, 'stages.3dm'), opts)
        print(f"[DEBUG] wrote stage geometry: {os.path.join(debug_dir, 'stages.3dm')}")
 
    options = Rhino.FileIO.FileWriteOptions()
    options.SuppressDialogBoxes = True
    ok = doc.WriteFile(out_step_path, options)
    if not ok:
        raise RuntimeError(f'Rhino failed to write STEP: {out_step_path}')
    print(f'Wrote STEP: {out_step_path}')
    return out_step_path
 
def stl_to_low_patch_nurbs_step(stl_path, step_path):
    """Convenience wrapper for the API-driven low-patch NURBS pipeline."""
    return mesh_to_low_patch_nurbs_step(
        stl_path,
        step_path,
        target_quads=1000,
        detect_hard_edges=False,
    )
 
if __name__ == "__main__":
    og_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\corner_optimisation\test"
    filename = "corner"
    morphed_no = "1"
    stl_file = os.path.join(og_path, f"{filename}_{morphed_no}.stl")
    fro_path = os.path.join(og_path, f"{filename}_{morphed_no}.fro")

    out_dir = os.path.join(og_path, "components")

    convert_fro_to_stl(fro_path, stl_file)

    comp_paths = save_components_as_stl(stl_file, out_dir)

    # # patch only component 1 first, likely the sphere
    # for i in range(len(comp_paths)):
    #     t1 = f"component_{i}.stl"
    #     path = os.path.join(out_dir,t1)
    #     stl_to_single_patch_step(
    #         path,
    #         os.path.join(out_dir, f"component_{i}_patch.stp"),
    #         max_points=3000
    #     )
        
        # component_mesh_to_nurbs_step(
        #     path,
        #     os.path.join(out_dir, f"component_{i}_mesh_nurbs.stp")
        # )
        
        # mesh_to_low_patch_nurbs_step(path, out_step_path=f"component_{i}_mesh_nurbs.stp", target_quads=1000, detect_hard_edges=False, save_debug_stages=True)