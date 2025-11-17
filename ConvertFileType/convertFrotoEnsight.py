import os
import sys

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("MeshQuality"))
from FileRW.FroFile import FroFile
from FileRW.Ensight import EnsightCase
from MeshQuality.NodeMetrics import average_node_spacing
from MeshQuality.FaceMetrics import face_aspect_ratio
from MeshQuality.FaceMetrics import f_size_3
from MeshQuality.FaceMetrics import f_skew_3
from MeshQuality.FaceMetrics import f_size_4
from MeshQuality.FaceMetrics import f_skew_4

def convert_fro_to_ensight(filepath, filename, case_name=None, quality=False, baseline=None):
    """
        filename should include fro extension (gets removed)
        if case_name is none, uses fro file name (minus extension)
    """
    if case_name == None:
        case_name = filename.split(".")[0]
    ff = FroFile.fromFile(f"{filepath}{filename}")
    ec = EnsightCase(case_name, filepath, ff)
    
    if quality:
        ff_baseline = None
        if baseline != None: 
            ff_baseline = FroFile.fromFile(baseline)
        
        ## Calculate and save average node spacing
        ff_ans  = average_node_spacing(ff, [])
        ec.scalars_per_node["spc"]     = ff_ans
        if baseline != None:
            ffb_ans = average_node_spacing(ff_baseline, [])
            abs_diffs = [abs(a-b) for a,b in zip(ff_ans, ffb_ans)]
            rel_diffs = [abs(a-b)/b for a,b in zip(ff_ans, ffb_ans)]
            ec.scalars_per_node["spc_abs"] = abs_diffs
            ec.scalars_per_node["spc_rel"] = rel_diffs
        
        # Calculate and save face aspect ratios
        ff_far  = face_aspect_ratio(ff, [])
        ec.scalars_per_elem["ar"]     = {"tria3" : ff_far}
        
        if baseline != None:
            ffb_far = face_aspect_ratio(ff_baseline, [])
            abs_diffs = [abs(a-b) for a,b in zip(ff_far, ffb_far)]
            rel_diffs = [abs(a-b)/b for a,b in zip(ff_far, ffb_far)]
            ec.scalars_per_elem["ar_abs"] = {"tria3" : abs_diffs}
            ec.scalars_per_elem["ar_rel"] = {"tria3" : rel_diffs}

        # Calculate and save relative size metrics as described in: 
        #   Knupp - Algebraic mesh quality metrics for unstructured initial meshes.
        # shape
        ff_fskw_3  = f_skew_3(ff, [])
        ff_fskw_4  = f_skew_4(ff, [])
        ec.scalars_per_elem["fskw"]     = {"tria3" : ff_fskw_3, "quad4" : ff_fskw_4}
        if baseline != None:
            ffb_fskw_3 = f_skew_3(ff_baseline, [])
            ffb_fskw_4 = f_skew_4(ff_baseline, [])
            abs_diffs_3 = [abs(a-b)   for a,b in zip(ff_fskw_3, ffb_fskw_3)]
            rel_diffs_3 = [abs(a-b)/b for a,b in zip(ff_fskw_3, ffb_fskw_3)]
            abs_diffs_4 = [abs(a-b)   for a,b in zip(ff_fskw_4, ffb_fskw_4)]
            rel_diffs_4 = [abs(a-b)/b for a,b in zip(ff_fskw_4, ffb_fskw_4)]
            ec.scalars_per_elem["fskw_abs"] = {"tria3" : abs_diffs_3, "quad4" : abs_diffs_4}
            ec.scalars_per_elem["fskw_rel"] = {"tria3" : rel_diffs_3, "quad4" : rel_diffs_4}
        ## size
        if baseline != None:
            ff_rsm_3  = f_size_3(ff, [], ff_baseline)
            ff_rsm_4  = f_size_4(ff, [], ff_baseline)
            ec.scalars_per_elem["rsm"] = {"tria3" : ff_rsm_3, "quad4" : ff_rsm_4}
            ff_fss_3 = [a * b for a,b in zip(ff_fskw_3, ff_rsm_3)]
            ff_fss_4 = [a * b for a,b in zip(ff_fskw_4, ff_rsm_4)]
            ec.scalars_per_elem["fss"] = {"tria3" : ff_fss_3, "quad4" : ff_fss_4}


    ec.write()
    
if __name__ == "__main__":
    filepath = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt\Outputs\l2"
    filename = "\morphedmesh.fro"
    convert_fro_to_ensight(filepath=filepath, filename=filename)