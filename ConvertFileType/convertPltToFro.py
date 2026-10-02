import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

for subdir in ["", "FileRW"]:
    p = os.path.join(project_root, subdir) if subdir else project_root
    if p not in sys.path:
        sys.path.insert(0, p)
        
from FileRW.FliteFile import FliteFile
from FileRW.Plt import Plt
from FileRW.FroFile import FroFile

def convert_plt_to_fro(filepath, filename):
    plt = Plt()
    print(f"reading file - {filepath}{filename}")
    plt.read_file(f"{filepath}{filename}")
    filename_short = filename.split(".")[0]
    plt.extract_fro_file(f"{filename_short}.fro")

if __name__ == "__main__":
    fn = FliteFile.getFileExtOptions("plt")
    convert_plt_to_fro(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\rec", fn)