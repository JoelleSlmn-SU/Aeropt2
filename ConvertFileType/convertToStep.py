import pyvista as pv
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.BRep import BRep_Builder
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.Geom import Geom_BSplineSurface
from OCP.IGESControl import IGESControl_Writer
from OCP.gp import gp_Pnt
from ShapeParameterization.surfaceFitting import fitSurfaceNURBs

def export_vtm_to_step(vtm_path, step_path):
    reader = pv.get_reader(vtm_path)
    multiblock = reader.read()

    faces = []
    for name in multiblock.keys():
        block = multiblock[name]
        if block is None or block.n_points < 9:
            continue
        try:
            # Fit surface and convert to OCC
            points, _, surf = fitSurfaceNURBs(block.points)
            ctrlpts2d = surf.ctrlpts2d
            if not ctrlpts2d:
                continue
            face = build_occ_face_from_nurbs(surf)
            faces.append(face)
        except Exception as e:
            print(f"Skipping surface '{name}': {e}")
            continue

    # Combine all faces into a single compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face)

    # Export to STEP
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(step_path)
    if status != 1:
        raise RuntimeError(f"STEP export failed with status {status}")

    print(f"STEP file written to: {step_path}")
    
def export_nurbs_to_step(surf, step_path):
    try:
        face = build_occ_face_from_nurbs(surf)
    except Exception as e:
        print(f"Failed to build OCC face from Nurbs: {e}")

    # Combine all faces into a single compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face)

    # Export to STEP
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(step_path)
    if status != 1:
        raise RuntimeError(f"STEP export failed with status {status}")

    print(f"STEP file written to: {step_path}")

def build_occ_face_from_nurbs(surf):
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.Geom import Geom_BSplineSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt

    ctrlpts2d = surf.ctrlpts2d
    num_u = len(ctrlpts2d)
    num_v = len(ctrlpts2d[0])
    poles = TColgp_Array2OfPnt(1, num_u, 1, num_v)

    for i in range(num_u):
        for j in range(num_v):
            x, y, z = ctrlpts2d[i][j]
            poles.SetValue(i+1, j+1, gp_Pnt(x, y, z))

    def build_knots_and_mults(knotvector, degree, tol=1e-6):
        unique_knots = []
        multiplicities = []
        prev = None
        count = 0
        for kv in knotvector:
            if prev is None or abs(kv - prev) > tol:
                if prev is not None:
                    multiplicities.append(count)
                unique_knots.append(kv)
                count = 1
                prev = kv
            else:
                count += 1
        multiplicities.append(count)

        kv = TColStd_Array1OfReal(1, len(unique_knots))
        mv = TColStd_Array1OfInteger(1, len(multiplicities))
        for i, val in enumerate(unique_knots):
            kv.SetValue(i+1, val)
            mv.SetValue(i+1, multiplicities[i])
        return kv, mv

    kv_u, mv_u = build_knots_and_mults(surf.knotvector_u, surf.degree_u)
    kv_v, mv_v = build_knots_and_mults(surf.knotvector_v, surf.degree_v)

    occ_surf = Geom_BSplineSurface(
        poles, kv_u, kv_v, mv_u, mv_v,
        surf.degree_u, surf.degree_v,
        False, False
    )

    face = BRepBuilderAPI_MakeFace(occ_surf, 1e-6).Face()
    return face

def export_nurbs_surface_to_step(surf, filepath, format='step'):
    """
    Converts a geomdl NURBS surface to a STEP or IGES file using OCP.
    Args:
        surf: geomdl.BSpline.Surface object
        filepath: output path (.step or .igs)
        format: 'step' or 'iges'
    """
    ctrlpts2d = surf.ctrlpts2d
    num_u = len(ctrlpts2d)
    num_v = len(ctrlpts2d[0])

    # Build OpenCASCADE-compatible control point array
    poles = TColgp_Array2OfPnt(1, num_u, 1, num_v)
    for i in range(num_u):
        for j in range(num_v):
            x, y, z = ctrlpts2d[i][j]
            poles.SetValue(i+1, j+1, gp_Pnt(x, y, z))

    # Knot vectors and multiplicities
    def build_knots_and_mults(knotvector, degree, tol=1e-6):
        unique_knots = []
        multiplicities = []

        prev = None
        count = 0
        for kv in knotvector:
            if prev is None or abs(kv - prev) > tol:
                if prev is not None:
                    multiplicities.append(count)
                unique_knots.append(kv)
                count = 1
                prev = kv
            else:
                count += 1
        multiplicities.append(count)

        kv = TColStd_Array1OfReal(1, len(unique_knots))
        mv = TColStd_Array1OfInteger(1, len(multiplicities))

        for i, val in enumerate(unique_knots):
            kv.SetValue(i+1, val)
            mv.SetValue(i+1, multiplicities[i])

        return kv, mv

    kv_u, mv_u = build_knots_and_mults(surf.knotvector_u, surf.degree_u)
    kv_v, mv_v = build_knots_and_mults(surf.knotvector_v, surf.degree_v)

    # Construct the OpenCASCADE B-spline surface
    occ_surf = Geom_BSplineSurface(
        poles, kv_u, kv_v, mv_u, mv_v,
        surf.degree_u, surf.degree_v,
        False, False
    )

    # Wrap as a face (required for export)
    face = BRepBuilderAPI_MakeFace(occ_surf, 1e-6).Face()

    if format.lower() == 'step':
        writer = STEPControl_Writer()
        Interface_Static.SetCVal_s("write.step.schema", "AP203")
        writer.Transfer(face, STEPControl_AsIs)
        status = writer.Write(filepath)
        if status != 1:
            raise RuntimeError(f"STEP export failed with status code {status}")
    elif format.lower() == 'iges':
        writer = IGESControl_Writer()
        writer.AddShape(face)
        status = writer.Write(filepath)
        if not status:
            raise RuntimeError("IGES export failed")
    else:
        raise ValueError("Format must be 'step' or 'iges'")
    
    print(f"NURBS surface exported to {filepath}")
