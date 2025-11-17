import os, sys
sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("ConvertFileType"))
from FileRW.BacFile import BacFile
from FileRW.BcoFile import BcoFile
from FileRW.FroFile import FroFile
from FileRW.DatFile import DatFile
from ConvertFileType.convertVtmtoFro import vtm_to_fro

if False:
    meshpath = os.path.join(os.getcwd(), "examples", "retry")
    focus = [94]
    bf = BacFile.fromMesh(
        name="cornerbump",
        mesh=os.path.join(meshpath, "CornerBump.fro"),
        mesh_path=meshpath,
        quality=90,
        focus_surfaces=focus
    )
    with open(os.path.join(meshpath, "CB.bac"), "w", newline="\n") as f:
        f.write(str(bf))
    print("OK -> CB.bac")

if False:
    filepath = os.path.join(os.getcwd(), "examples", "AirbusGeo")
    fro_path = os.path.join(filepath, "CornerBump.fro")

    ff = FroFile()
    ff.read_file(fro_path)

    bco = BcoFile.fromFro(ff, name="CornerBump")
    with open(os.path.join(filepath, "CornerBump.bco"), "w", newline="\n") as f:
        f.write(str(bco))
        
if True:
    df = DatFile(os.path.join(os.getcwd(), "examples/actualfinal", "CornerBump.dat"))
    '''stats = df.clean(
        tol=1e-6,
        mode="euclid",
        dedupe=False,
        sig_decimals=6,
        short_tol=0,
        rel_factor=0,
        # NEW:
        relink_surfaces=None,
        endpoint_tol=1e-10,     # tighten/loosen to taste
        len_rel_tol=0.05,      # ±5% length match
        out_path=os.path.join(os.getcwd(), "examples/actualfinal", "CornerBump_clean.dat")
    )
    print(stats)'''
    
    stats = df.clean_manual(
        out_path=os.path.join(os.getcwd(), "examples/actualfinal", "CornerBumpFIXED.dat"),
        remove_surfaces=[],
        remove_curves=[246,316]
    )
    print(stats)
    
if False:
    df = DatFile(os.path.join(os.getcwd(), "examples/retry", "CornerBump.dat"))
    
    stats = df.convert_units(
        out_path=os.path.join(os.getcwd(), "examples/retry", "CornerBump_converted.dat"),
        from_unit="mm",
        to_unit="m",
        zero_threshold=1e-8
    )
    print(stats)

if False:
    ff = FroFile.fromFile(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\AirbusGeoFro\CornerBump.fro")

    # Clean with duplicate + unused removal
    ff.clean(tol=1e-5, remove_unreferenced=True)

    ff.write_file(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\AirbusGeoFro\CornerBump_clean.fro")
    