import json
import os, sys
import posixpath

# ------------------------------------------------------------------
# Resolve project root robustly and add paths so FileRW is importable
# ------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = script_dir
while True:
    # We want the directory that *contains* FileRW
    if os.path.isdir(os.path.join(project_root, "FileRW")):
        break
    parent = os.path.dirname(project_root)
    if parent == project_root:
        # Reached filesystem root; give up
        break
    project_root = parent

# Put the project root on sys.path if we found FileRW
if os.path.isdir(os.path.join(project_root, "FileRW")):
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "ConvertFileType"))
    sys.path.insert(0, os.path.join(project_root, "MeshGeneration"))
else:
    # Fallback: at least add script_dir so relative imports still have a chance
    sys.path.insert(0, script_dir)

from FileRW.FroFile import FroFile
from ConvertFileType.convertVtmtoFro import vtm_to_fro
from ConvertFileType.convertEnGeoToFro import engeo_to_fro
from MeshGeneration.MorphModel import MorphModel
from MeshGeneration.Morph import MorphMesh


def _expand(p: str) -> str:
    # Expand $HOME, $VARS and ~
    if not isinstance(p, str):
        return p
    p = os.path.expandvars(p)
    p = os.path.expanduser(p)
    return p

def run_surf_morph_from_config(fro_target_path, json_path):
    import numpy as np

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_path, "r") as f:
        config = json.load(f)

    out_dir   = _expand(config["output_directory"])# e.g. /home/.../aeropt/aeropt_out/<run>/ 
    n         = int(config["n"])
    gen       = int(config["gen"])
    vtk_name  = config["vtk_name"]                      # uploaded filename (with extension)
    filetype  = config["mesh filetype"]                 # ".vtk" or ".vtm" or ".fro"
    debug     = bool(config.get("debug", False))
    rigid_translation = config["rigid_translation"]

    base_name = os.path.basename(vtk_name).rsplit(".", 1)[0]
    
    # Remote directory containing the uploaded mesh and where we will write crm.fro
    remote_dir = posixpath.join(out_dir, "surfaces", f"n_{gen}")
    os.makedirs(remote_dir, exist_ok=True)

    # Build morph model from JSON
    morpher = MorphModel(
        f_name=None, con=False, pb=[], path=None,
        T=[], U=[],
        cn=np.asarray(config["control_nodes"], float).tolist(),
        displacement_vector=np.asarray(config["displacement_vector"], float).tolist()
    )
    morpher.t_surfaces = list(map(int, config["t_surfaces"]))
    morpher.u_surfaces = list(map(int, config["u_surfaces"]))
    morpher.c_surfaces = list(map(int, config["c_surfaces"]))
    morpher.rigid_boundary_translation = rigid_translation

    # Ensure we have a FroFile ready
    fro_path = f"{base_name}.fro"  # where we will create or read the fro
    if filetype in (".vtk", ".vtm"):
        vtk_path = posixpath.join(remote_dir, vtk_name)
        if not os.path.exists(vtk_path):
            raise FileNotFoundError(f"Remote mesh not found: {vtk_path}")
        # Convert uploaded VTK/VTM -> FRO at the specified fro_path
        print(f"[REMOTE MORPH] Converting {vtk_path} -> {fro_path}")
        m_0 = vtm_to_fro(vtk_path, fro_path)
    elif filetype in (".case"):
        case_path = posixpath.join(remote_dir, vtk_name)
        if not os.path.exists(case_path):
            raise FileNotFoundError(f"Remote mesh not found: {case_path}")
        # Convert uploaded VTK/VTM -> FRO at the specified fro_path
        print(f"[REMOTE MORPH] Converting {case_path} -> {fro_path}")
        m_0 = engeo_to_fro(case_path, fro_path)
    elif filetype == ".fro":
        if not os.path.exists(fro_path):
            raise FileNotFoundError(f"Expected .fro path not found: {fro_path}")
        m_0 = FroFile(fro_path)
    else:
        raise ValueError(f"Unsupported mesh filetype: {filetype}")

    
    if m_0.node_count == 0 or (len(m_0.boundary_triangles) + len(m_0.boundary_quads)) == 0:
        raise RuntimeError(
            f"Converted .fro is empty. Check that '{vtk_path}' and its piece files exist. "
            f"Expected sidecar dir next to the VTM (e.g., {os.path.splitext(vtk_path)[0]}/)."
        )
    morphed = MorphMesh(m_0, base_name, morpher, viewer=None, output_dir=out_dir, debug=debug, rb="original", n=n, gen=gen)

    out_path = posixpath.join(remote_dir, f"{base_name}_{n}.fro")
    morphed.write_file(out_path)
    print(f"[REMOTE MORPH] Morphed file saved to: {out_path}")
    return out_path

def run_cad_morph_from_config(step_target_path, json_path):
    """
    Placeholder for future CAD-FFD morph.

    When we wire CAD morphing through HPC, this function will:
      - load the CAD (STEP/IGES),
      - apply the FFD lattice & control-node displacements,
      - write the morphed CAD back to step_target_path.
    For now it just raises to make its use explicit.
    """
    raise NotImplementedError("CAD morph via remoteMorph.py is not implemented yet.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remoteMorph.py <target_path> <morph_config.json>")
        sys.exit(1)

    target_path = sys.argv[1]
    json_path   = sys.argv[2]

    # Try to detect whether this is a CAD morph from the config
    morph_kind = "mesh"
    try:
        with open(json_path, "r") as f:
            cfg = json.load(f)
        morph_kind = cfg.get("morph_kind", "mesh")
    except Exception:
        pass

    if morph_kind == "cad":
        run_cad_morph_from_config(target_path, json_path)
    else:
        run_surf_morph_from_config(target_path, json_path)