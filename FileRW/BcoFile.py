import os, sys
sys.path.append(os.path.dirname("FileRW"))
from FileRW.FliteFile import FliteFile

class BcoFile(FliteFile):
    def __init__(self, name="default"):
        super().__init__(ext="bco", exc_p=[], exc_f=[])
        self.filename = f"{name}.bco"
        self.surfaces = {}
        self.segments = {}

    @classmethod
    def default(cls, nsur:int, nseg:int, ffs:list, name:str="default"):
        """
            This is the default constructor. Use this to replace any BcoFile(...) instantiations.
        """
        bf = cls(name)
        for i in range(0, nsur):
            bf.set_surface(i+1, 1)

        for i in range(0, nseg):
            bf.set_segment(i+1, 0)

        for ff in ffs:
            bf.set_surface(ff, 3)
        
        return bf

    @classmethod
    def defaultCRM(cls):
        """
            If in doubt and need one for skylon, use this.
        """
        bf = cls.default(83,10,"Skylon")
        
        for ff in [15,16]:
            bf.set_surface(ff, 3)
        return bf

    @classmethod
    def fromLines(cls, name:str, lines: list):
        """
            Constructs a BcoFile from the result of file.readlines()
        """
        line = 0
        # ignore sur seg header line
        line += 1
        # read values for sur seg
        surseg = lines[line].split()
        surfaces = int(surseg[0])
        segments = int(surseg[1])
        
        bf = cls.default(surfaces, segments, name)

        line += 1
        # ignore surface header line
        line += 1
        # read values for surfaces
        for i in range(line, surfaces+line):
            vals = lines[line].split()
            line += 1
            bf.set_surface(vals[0], vals[1])
        # ignore segment header line
        line += 1
        # read values for segment
        for i in range(line, segments+line):
            vals = lines[line].split()
            line += 1
            bf.set_segment(vals[0], vals[1])
        return bf

    @classmethod
    def fromFile(cls, filepath: str):
        name = os.path.basename(filepath)[:-4]
        lines = open(filepath).readlines()
        # basic sanity: first header should contain these tokens
        if not lines or "nsur" not in lines[0].lower() or "nsegm" not in lines[0].lower():
            raise ValueError(f"{filepath} does not look like a .bco file")
        return cls.fromLines(name, lines)
    
    @classmethod
    def fromFro(cls, fro_obj, name="default", enabled=None):
        nsur = len(fro_obj.get_surface_ids())
        bco = cls.default(nsur, 0, [], name)     # start with all surfaces flag=1
        enabled = set(enabled or [])
        for sid in fro_obj.get_surface_ids():
            bco.set_surface(int(sid), 3 if int(sid) in enabled else 1)
        return bco
        
    @classmethod
    def fromString(cls, name:str, data: str):
        """
            data coming in from the server is sometimes gathered using cat <filename> and so is handed over to BcoFile as a string
        """
        # this class method splits that data into the inidiviual lines used to initialise the class
        file = data.split("\n")
        return cls.fromLines(name, file)

    @classmethod
    def local(cls):
        """
            looks for in CWD a .bco file and creates BcoFile instance from it. Will request user input if multiple found.
        """
        filename = cls.getFileExtension("bco")
        if filename is None:
            return None
        return cls.fromFile(filename)

    def set_surface(self, id:int, flag:int) -> None:
        """
            TODO - List flag options here
        """
        self.surfaces[str(id)] = str(flag)
    
    def set_segment(self, id:int, flag:int) -> None:
        """
            TODO - List flag options here
        """
        self.segments[str(id)] = str(flag)

    def surface_num(self) -> int:
        """
            Returns the number of surfaces currently in the BcoFile object.
        """
        return len(self.surfaces.keys())

    def segment_num(self) -> int:
        """
            Returns the number of segments currently in the BcoFile object.
        """
        return len(self.segments.keys())

    def __str__(self) -> str:
        retval = ""
        retval += "nsur    nsegm\n"
        retval += f"{str(len(self.surfaces.keys())): >9}{str(len(self.segments.keys())): >9}\n"
        retval += "surfaces\n"
        for key in self.surfaces.keys():
            retval += f"{str(key): >9}{str(self.surfaces[key]): >9}{str(0): >9}\n"
        retval += "segments\n"
        for key in self.segments.keys():
            retval += f"{str(key): >9}{str(self.segments[key]): >9}{str(0): >9}\n"
        retval += "moving surfaces\n"
        return retval

if __name__ == "__main__":
    bco = BcoFile.local()
    if bco is None:
        print("No BCO file found in local directory")
    else:
        os.system(f"echo '{bco}' | less -f /dev/stdin")