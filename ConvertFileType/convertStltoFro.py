import os, sys
sys.path.append(os.path.dirname("FileRW"))
from FileRW.StlFile import Stl
from FileRW.FroFile import FroFile
from FileRW.FliteFile import FliteFile

def convert_stl_to_fro(filepath):
    filepath = os.path.normpath(filepath)
    stl = Stl()
    stl.read_file(filename=filepath)
    fro = FroFile.fromMesh(stl)
    fro.write_file(f"{filepath[:-4]}.fro")
    return fro

if __name__ == "__main__":
    fn = FliteFile.getFileExtOptions("stl")
    convert_stl_to_fro(fn)