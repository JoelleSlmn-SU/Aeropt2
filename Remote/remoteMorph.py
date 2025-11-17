import json
import os, sys
import posixpath

# Resolve project root and add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # .../Scripts
sys.path.insert(0, project_root)
sys.path.insert(0, posixpath.join(project_root, "FileRW"))
sys.path.insert(0, posixpath.join(project_root, "ConvertFileType"))
sys.path.insert(0, posixpath.join(project_root, "MeshGeneration"))

from FileRW.FroFile import FroFile
from ConvertFileType.convertVtmtoFro import vtm_to_fro
from ConvertFileType.convertEnGeoToFro import engeo_to_fro
from MeshGeneration.MorphModel import MorphModel
from MeshGeneration.MorphBS import MorphMesh

def _expand(p: str) -> str:
    # Expand $HOME, $VARS and ~
    if not isinstance(p, str):
        return p
    p = os.path.expandvars(p)
    p = os.path.expanduser(p)
    return p

def run_morph_from_config(fro_target_path, json_path):
    import numpy as np

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_path, "r") as f:
        config = json.load(f)

    out_dir  = _expand(config["output_directory"])# e.g. /home/.../aeropt/aeropt_out/<run>/ 
    n         = int(config["n"])
    vtk_name  = config["vtk_name"]                      # uploaded filename (with extension)
    filetype  = config["mesh filetype"]                 # ".vtk" or ".vtm" or ".fro"
    debug     = bool(config.get("debug", False))

    base_name = os.path.basename(vtk_name).rsplit(".", 1)[0]
    
    # Remote directory containing the uploaded mesh and where we will write crm.fro
    remote_dir = posixpath.join(out_dir, "surfaces", f"n_{n}")
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

    # Ensure we have a FroFile ready
    fro_path = _expand(fro_target_path)  # where we will create or read the fro
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
    morphed = MorphMesh(m_0, base_name, morpher, viewer=None, output_dir=out_dir, debug=debug, rb="original")

    out_path = posixpath.join(remote_dir, f"{base_name}.fro")
    morphed.write_file(out_path)
    print(f"[REMOTE MORPH] Morphed file saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remoteMorph.py <target.fro> <morph_config.json>")
        sys.exit(1)
    run_morph_from_config(sys.argv[1], sys.argv[2])
