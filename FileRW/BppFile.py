import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.FliteFile import FliteFile

class BppFile(FliteFile):
    def __init__(self, name="default"):
        self.filename = "{}.bpp".format(name)
        self.Globe_GridSize         = []
        self.Curvature_Factors      = []
        self.Layer_Heights          = []
        self.UnChecking_Curve_IDs   = []
        self.UnChecking_Surf_IDs    = []
        self.Symmetry_Surf_IDs      = []
        self.FarField_Surf_IDs      = []
        self.UnChecking_Curve_Names = []
        self.UnChecking_Surf_Names  = []
        self.Symmetry_Surf_Names    = []
        self.FarField_Surf_Names    = []


    @classmethod
    def defaultCRM(cls):
        bf = cls("CRM")
        bf.Globe_GridSize = [100]
        bf.Curvature_Factors = [0.15, 0.01, 0.005, 0.1, 120.0, 90.0, 30.0]
        bf.Layer_Heights = [0.004,	0.0048,	0.00576,	0.006912,	0.0082944,	0.00995328,	
                        0.011943936,	0.014332723,	0.017199268,	0.020639121,	0.024766946,	0.029720335,	0.035664402,	
                        0.042797282,	0.051356739,	0.061628086,	0.073953704,	0.088744444,	0.106493333,	0.127792,	
                        0.1533504,	0.18402048,	0.220824576,	0.264989491,	0.317987389]
        bf.FarField_Surf_IDs = [15, 16]
        return bf

    @classmethod
    def fromLines(cls, name: str, lines: list):
        bf = cls(name)
        section = None
        layer_heights_num_ignored = False
        count = 0
        for line in lines:
            line = line.strip()
            count += 1
            try:
                if line.isspace() or line == "" and section != None:
                    setattr(bf, section, line.split())
                    section = None
                elif line.isspace() or line == "":
                    continue
                elif line[0] == "#":
                    section = line.replace("#", "").strip()
                elif section == "Layer_Heights":
                    if layer_heights_num_ignored:
                        vals = line.split()
                        setattr(bf, section, vals)  
                        section = None
                    else: 
                        layer_heights_num_ignored = True
                        if line[0] == "0":
                            vals = []
                            setattr(bf, section, vals)
                            section = None
                else:
                    setattr(bf, section, line.split())
                    section = None
            except TypeError as tex:
                print(f"ERROR - Could not parse line {count} in {bf.filename}")
                if count > 1:
                    print(f"Previous: {lines[count-2]}")
                print(f"Current : {lines[count-1]}")
                if count < len(lines):
                    print(f"Next    : {lines[count]}")
        return bf

    @classmethod
    def fromFile(cls, filepath: str):
        # sometimes the bac file needs to be initialsed from a filepath 
        name = filepath.split("/")[-1][:-4] # replace with os.sep?
        file = open(filepath).readlines()
        return cls.fromLines(name, file)

    @classmethod
    def fromDict(cls, name: str, dictionary: dict):
        bf = cls(name)
        for param in dictionary["BPP Params"]:
            ints = []
            for par in param[1].split():
                try:
                    num = int(par)
                    ints.append(num)
                except:
                    try:
                        num = float(par)
                        ints.append(num)
                    except:
                        print("major panik - real")
                
            setattr(bf, param[0], ints)
        return bf
                
    @classmethod
    def local(cls):
        filename = cls.getFileExtension("bpp")
        if filename is None:
            return None
        return cls.fromFile(filename) 

    def __str__(self) -> str:
        retval = "#Unchecking_Curve_IDs\n"
        for num in self.UnChecking_Curve_IDs:
            retval += f"{num} "
        retval += "\n"
        retval += "#UnChecking_Surf_IDs\n"
        for num in self.UnChecking_Surf_IDs:
            retval += f"{num} "
        retval += "\n"
        retval += "#Highorder_Surf_IDs\n"
        retval += "\n"
        retval += "\n"
        retval += "#Generating_Surf_IDs\n"
        retval += "\n"
        retval += "\n"
        retval += "#Unchecking_Curve_Names\n"
        for num in self.UnChecking_Curve_Names:
            retval += f"{num} "
        retval += "\n"
        retval += "#Unchecking_Surf_Names\n"
        for num in self.UnChecking_Surf_Names:
            retval += f"{num} "
        retval += "\n"
        retval += "#Generating_Surf_Names\n"
        retval += "\n"
        retval += "\n"
        retval += "#Layer_Heights\n"
        retval += f"{len(self.Layer_Heights)}\n"
        if len(self.Layer_Heights) > 0:
            for num in self.Layer_Heights:
                retval += f" {float(num):.10f}".rstrip("0")
            retval += "\n"
        retval += "\n"
        retval += "#SuperPatch_Groups\n"
        retval += " 0"
        retval += "\n"
        retval += "#SuperCurve_RidgeAngle\n"
        retval += " 150"
        retval += "\n"
        retval += "#Symmetry_Surf_IDs\n"
        for num in self.Symmetry_Surf_IDs:
            retval += f" {num}"
        retval += "\n"
        retval += "#farfield_Surf_IDs\n"
        if len(self.FarField_Surf_IDs) > 0:
            for num in self.FarField_Surf_IDs:
                retval += f" {num}"
            retval += "\n"
        retval += "#Symmetry_Surf_Names\n"
        for num in self.Symmetry_Surf_Names:
            retval += f" {num}"
        retval += "\n"
        retval += "#FarField_Surf_Names\n"
        for num in self.FarField_Surf_Names:
            retval += f" {num}"
        retval += "\n"
        retval += "#end\n"
        return retval

    def to_dictionary(self) -> dict:
        def list_to_string(in_list):
            retval = ""
            for x in in_list:
                retval += str(x) + " "
            return retval
        
        params = {}
        params["BPP Params"] = []
        params["BPP Params"].append(("Globe_GridSize",         list_to_string(self.Globe_GridSize)))
        params["BPP Params"].append(("Curvature_Factors",      list_to_string(self.Curvature_Factors)))
        params["BPP Params"].append(("Layer_Heights",          list_to_string(self.Layer_Heights)))
        params["BPP Params"].append(("UnChecking_Curve_IDs",   list_to_string(self.UnChecking_Curve_IDs)))
        params["BPP Params"].append(("UnChecking_Surf_IDs",    list_to_string(self.UnChecking_Surf_IDs)))
        params["BPP Params"].append(("Symmetry_Surf_IDs",      list_to_string(self.Symmetry_Surf_IDs)))
        params["BPP Params"].append(("FarField_Surf_IDs",      list_to_string(self.FarField_Surf_IDs)))
        params["BPP Params"].append(("UnChecking_Curve_Names", list_to_string(self.UnChecking_Curve_Names)))
        params["BPP Params"].append(("UnChecking_Surf_Names",  list_to_string(self.UnChecking_Surf_Names)))
        params["BPP Params"].append(("Symmetry_Surf_Names",    list_to_string(self.Symmetry_Surf_Names)))
        params["BPP Params"].append(("FarField_Surf_Names",    list_to_string(self.FarField_Surf_Names)))
        return params

if __name__ == "__main__":
    bpp = BppFile.local()
    if bpp is None:
        print("No BPP file found in local directory")
    else:
        os.system(f"echo '{bpp}' | less -f /dev/stdin")