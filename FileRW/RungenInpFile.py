from datetime import datetime
import os
from FileRW.BcoFile import BcoFile

class RungenInpFile():
    def __init__(self, project="default", file=None, invvis=2, number_of_grids=1, hybrid=False, parallel_domains=120, roll_ground=False, ground_angle=45, starting_step_in_cycle=1, steps_per_cycle=1):
        self.project = project
        self.filename = "rungen.inp"
        self.invvis = invvis
        self.number_of_grids = number_of_grids
        self.hybrid = hybrid
        self.parallel_domains = parallel_domains
        self.roll_ground = roll_ground
        self.ground_angle = ground_angle
        self.starting_step_in_cycle = starting_step_in_cycle
        self.steps_per_cycle = steps_per_cycle

        if file:
            self.parse_from_file(file)

    def parse_from_file(self, lines):
        line = 0
        self.project = lines[line]
        line += 1
        self.invvis = int(lines[line])
        line += 1
        self.number_of_grids = int(lines[line])
        line += 1
        self.hybrid = self.char_to_bool(lines[line])
        line += 1
        self.parallel_domains = int(lines[line])
        line += 1
        self.roll_ground = self.char_to_bool(lines[line])
        line += 1
        if self.roll_ground:
            self.ground_angle = int(lines[line])
            line += 1
        self.starting_step_in_cycle = int(lines[line])
        line += 1
        self.steps_per_cycle = int(lines[line])
        line += 1

    def __str__(self):
        retval = ""
        retval += "{}\n".format(self.project)
        retval += "{}\n".format(self.invvis)
        retval += "{}\n".format(self.number_of_grids)
        retval += "{}\n".format(self.bool_to_char(self.hybrid))
        retval += "{}\n".format(self.parallel_domains)
        retval += "{}\n".format(self.bool_to_char(self.roll_ground))
        if self.roll_ground:
            retval += "{}\n".format(self.ground_angle)
        retval += "{}\n".format(self.starting_step_in_cycle)
        retval += "{}\n".format(self.steps_per_cycle)
        return retval

    @classmethod
    def from_template(cls, template_path: str, project: str):
        """
        Load a template rungen.inp (if provided), override the first line with project name,
        and parse the remainder. If template missing, return sensible defaults.
        """
        inst = cls(project=project)
        try:
            if template_path and os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip() != ""]
                if lines:
                    lines[0] = project
                    inst.parse_from_file(lines)
        except Exception:
            # fall through to defaults
            pass
        return inst
    
    @classmethod
    def read(cls, path: str):
        """Read rungen.inp from a file path and return a populated instance."""
        inst = cls()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip() != ""]
        inst.parse_from_file(lines)
        return inst

    def write(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(self))
        return out_path

    def bool_to_char(self, val):
        if val:
            return 't'
        else:
            return 'f'

    def char_to_bool(self, val):
        val = val.strip().strip("\n").strip()
        if val == "t" or val == "True":
            return True
        else:
            return False

    def parse_from_dictionary(self, params):
        self.invvis = int(params["inputs"][0][1])
        self.number_of_grids = int(params["inputs"][1][1])
        self.hybrid = self.char_to_bool(params["inputs"][2][1])
        self.parallel_domains = int(params["inputs"][3][1])
        self.roll_ground = self.char_to_bool(params["inputs"][4][1])
        self.ground_angle = int(params["inputs"][5][1])
        self.starting_step_in_cycle = int(params["inputs"][6][1])
        self.steps_per_cycle = int(params["inputs"][7][1])

    def to_dictionary(self):
        params = {}
        params["inputs"] = []
        params["inputs"].append(("Inviscid (1) or Viscous (2)",self.invvis))
        params["inputs"].append(("Number of Grids",self.number_of_grids))
        params["inputs"].append(("Is Hybrid Mesh (t/f)",self.hybrid))
        params["inputs"].append(("Parallel Domains",self.parallel_domains))
        params["inputs"].append(("Rolling Ground (t/f)",self.roll_ground))
        params["inputs"].append(("Ground Angle (deg)",self.ground_angle))
        params["inputs"].append(("Starting Step In Cycle",self.starting_step_in_cycle))
        params["inputs"].append(("Steps per Cycle",self.steps_per_cycle))
        return params

if __name__ == "__main__":
    rg = RungenInpFile("Skylon")
    print(rg)