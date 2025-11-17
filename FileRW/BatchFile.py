import os
from FileRW.FliteFile import FliteFile

class Batchfile(FliteFile):
    def __init__(self, jobname="default"):
        self.sbatch_params = {}
        self.sbatch_params["job-name"] = jobname
        self.sbatch_params["output"] = "bench.out.%J"
        self.sbatch_params["error"] = "bench.err.%J"
        self.sbatch_params["time"] = "0-20:00"
        self.sbatch_params["ntasks"] = "1"
        #self.sbatch_params["mem-per-cpu"] = "4000"
        self.sbatch_params["ntasks-per-node"] = "1"
        
        self.lines = []

    def __str__(self):
        retval = ""

        retval += "#!/bin/bash --login\n"
        retval += "###\n"

        for key in self.sbatch_params.keys():
            retval += f"#SBATCH --{key}={self.sbatch_params[key]}\n"
        retval += "###\n"
        retval += "\n"

        for line in self.lines:
            retval += f"{line}\n"

        return retval

    @classmethod
    def fromFile(cls, filename):
        lines = open(filename).readlines()
        return cls.fromLines(lines)

    @classmethod
    def fromLines(cls, lines):
        bf = cls()
        for i in range(len(lines)):
            lines[i] = lines[i].strip().strip("\n").strip()
        line = 0
        
        assert lines[line] == "#!/bin/bash --login"
        line += 1 # discard first line as it is just #!/bin/bash --login

        assert lines[line] == "###"
        line += 1 # discard second line as it is just ###

        while lines[line] != "###":
            # loop through SBATCH params 
            k,v = lines[line].split("--")[1].split("=")
            bf.sbatch_params[k] = v
            line += 1
        
        assert lines[line] == "###"
        line += 1 # discard line as it is just ###
        bf.lines = lines[line:]
        return bf

    @classmethod
    def local(cls):
        filename = cls.getMatch("batchfile")
        if filename is None:
            return None
        return cls.fromFile(filename) 

if __name__ == "__main__":
    bat = Batchfile.local()
    if bat is None:
        print("No batch file found in local directory")
    else:
        os.system(f"echo '{bat}' | less -f /dev/stdin")
    #testnum = 1
    #src_dir = "srcdir/"
    #output_dir = "surfacemesh"
    #bf = Batchfile(f"doe{testnum}")

    #bf.lines.append(f"cd {output_dir}")
    #bf.lines.append(f"python3 -u {src_dir}doe.py run {testnum}")
    #bf.lines.append(f"")
    #bf.lines.append(f"module load intel")
    #bf.lines.append(f"")
    #bf.lines.append(f"mesh_quality > mesh_quality.log <<INPUT1")
    #bf.lines.append(f"{src_dir}Skylon.fro")
    #bf.lines.append(f"INPUT1")
    #bf.lines.append(f"")

    #print(bf)
