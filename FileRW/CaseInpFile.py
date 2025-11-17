import os 
from FileRW.ConfigFiles import config_folder

class CaseInpFile():
    def __init__(self, name="default", file=None, filepath=None):
        self.filename = f"{name}.inp"
        self.key_params = {}
        self.engine_params = {}
        self.other_params = {}

        self.val_strings = ["solFilesDirectory", "dataDirectory"]
        self.bool_strings= ["GUSTMODEL", "SAS", "useDissipationWeighting", "explicit",
                            "useMatrixDissipation", "patchInitialization", "movingwall", "wallsAreIsentropic"]

        # ---- SAFE defaults (won't crash if Case.inp is absent)
        self.parse_defaults()

        # If a concrete file path is provided, parse it (overrides defaults)
        if filepath and os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            self.source_lines = list(lines)  # <-- preserve for write_preserving_source
            self.parse_from_file(lines)

    def parse_from_file(self, file):
        for line in file:
            if line.isspace() or line == "":
                continue
            elif line[0] == "/":
                continue
            elif line[0] == "&":
                continue
            else:
                k,v = self.parse_line(line)
                if k in self.engine_params.keys():
                    self.engine_params[k] = v
                elif k in self.key_params.keys():
                    self.key_params[k] = v
                else:
                    self.other_params[k] = v

    def parse_defaults(self):
        # Try a few sensible locations; if none exist, just keep empty dicts
        candidates = [
            os.path.join(config_folder(), "Case.inp"),
            os.path.join(os.getcwd(), "templates", "Case.inp"),
            os.path.join(os.path.dirname(__file__), "..", "ConfigFiles", "Case.inp"),
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if not src:
            # No defaults found → leave dicts empty and return silently
            return

        with open(src, "r", encoding="utf-8", errors="ignore") as default_file:
            lines = default_file.readlines()

        for line in lines:
            if line.isspace() or line == "":
                continue
            elif line[0] in ("/", "&"):  # skip comments/sections you already skip
                continue
            elif line[:2] == "!!":
                kv = self.parse_line(line)
                self.engine_params[kv[0]] = kv[1]
            elif line[0] == "!":
                kv = self.parse_line(line)
                self.key_params[kv[0]] = kv[1]
            else:
                kv = self.parse_line(line)
                self.other_params[kv[0]] = kv[1]

    def parse_line(self, line):
        try:
            line = line.replace(" ", "").strip("\n").strip("!").strip(",").split("%")[1]
        except IndexError as iex:
            print(line)
            raise
        kv = line.split("=")
        k = kv[0]
        v = kv[1]
        if v == ".true.":
            return (k, True)
        elif v == ".false.":
            return (k, False)
        elif "'" in v:
            return (k,v)
        elif "." in v or "e" in v:
            return (k, float(v))
        else:
            return (k, int(v))

    def parse_from_dictionary(self, params):
        for kv in params["Key"]:
            self.key_params[kv[0]] = kv[1]
        for kv in params["Engine"]:
            self.engine_params[kv[0]] = kv[1]
        for kv in params["Other"]:
            self.other_params[kv[0]] = kv[1]

    def __str__(self):
        retval = ""
        retval += "&inputVariables\n"
        for key in self.key_params.keys():
            if key in self.val_strings:
                retval += "\tivd%{} = '{}',\n".format(key, self.key_params[key].strip().strip("'"))
            elif str(self.key_params[key]) == "True":
                    retval += "\tivd%{} = {},\n".format(key, ".true.")
            elif str(self.key_params[key]) == "False":
                retval += "\tivd%{} = {},\n".format(key, ".false.")
            else:
                retval += "\tivd%{} = {},\n".format(key, self.key_params[key])
        retval += "\n"
        for key in self.engine_params.keys():
            retval += "\tivd%{} = {},\n".format(key, self.engine_params[key])
        retval += "\n"
        for key in self.other_params.keys():
            if str(self.other_params[key]) == "True":
                    retval += "\tivd%{} = {},\n".format(key, ".true.")
            elif str(self.other_params[key]) == "False":
                retval += "\tivd%{} = {},\n".format(key, ".false.")
            else:
                retval += "\tivd%{} = {},\n".format(key, self.other_params[key])
        retval.strip(",")
        retval += "/"
        return retval

    def _fmt_value(self, key, val):
        # booleans -> .true./.false.
        if isinstance(val, bool):
            return ".true." if val else ".false."
        # string keys that require quotes
        if key in getattr(self, "val_strings", []):
            s = str(val).strip().strip("'")
            return f"'{s}'"
        # numeric -> print as is (avoid scientific unless already floaty)
        return str(val)

    def write_preserving_source(self, out_path: str):
        import re
        """
        If this instance was built from a filepath/template, rewrite that exact
        file shape and only replace values for keys we know about. Any keys present
        in self.*_params but missing from the template are appended right before '/'.
        """
        # 1) Load the source lines (we prefer the same file we parsed; if not, synthesize from __str__)
        if hasattr(self, "source_lines") and self.source_lines:
            lines = list(self.source_lines)
        else:
            # fallback: build a synthetic file from current dicts
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(str(self))
            return

        # 2) Collect all params into a flat dict
        full = {}
        full.update(self.key_params)
        full.update(self.engine_params)
        full.update(self.other_params)

        seen = set()

        def _patch_line(line: str) -> str:
            # match ivd%Key = value [with optional comma]
            m = re.match(r"(\s*ivd%)([A-Za-z0-9_()\[\]]+)(\s*=\s*)([^,/\n]+)(,?\s*)(.*)", line)
            if not m:
                return line
            key = m.group(2)
            if key not in full:
                return line
            seen.add(key)
            newv = self._fmt_value(key, full[key])
            # preserve comma if it existed
            return f"{m.group(1)}{key}{m.group(3)}{newv}{m.group(5)}{m.group(6)}"

        # 3) Patch lines in-place, remember where to insert new keys (before a line that is just '/')
        out_lines = []
        insert_idx = None
        for i, ln in enumerate(lines):
            if ln.strip() == "/":
                insert_idx = len(out_lines)  # insert before this
            out_lines.append(_patch_line(ln))

        # 4) Append any missing keys in a nice way (before '/')
        missing = [k for k in full.keys() if k not in seen]
        if missing:
            ins = insert_idx if insert_idx is not None else len(out_lines)
            add = []
            for k in missing:
                v = self._fmt_value(k, full[k])
                add.append(f"\tivd%{k} = {v},\n")
            out_lines[ins:ins] = add  # insert before '/'

        # 5) Ensure file ends with '/' (keeps template’s final slash if present)
        if not any(ln.strip() == "/" for ln in out_lines):
            out_lines.append("/\n")

        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(out_lines)
    
    
    def to_dictionary(self):
        params = {}
        params["Key"] = []
        params["Engine"] = []
        params["Other"] = []
        for key in self.key_params.keys():
            params["Key"].append((key, self.key_params[key]))
        for key in self.engine_params.keys():
            params["Engine"].append((key, self.engine_params[key]))
        for key in self.other_params.keys():
            params["Other"].append((key, self.other_params[key]))
        return params

    def get_param(self, key):
        if key in self.key_params:
            return self.key_params[key]
        if key in self.engine_params:
            return self.engine_params[key]
        if key in self.other_params:
            return self.other_params[key]
        return None
    
    def set_param(self, key, value):
        if key in self.key_params:
            self.key_params[key] = value
        if key in self.engine_params:
            self.engine_params[key] = value
        if key in self.other_params:
            self.other_params[key] = value

    def get_diffs(self, other):
        
        def clean(val):
            return str(val).strip()
        
        differences = []
        missing     = []

        for k, v in self.key_params.items():
            if k == "solFilesDirectory" or k == "dataDirectory":
                continue
            if k not in other.key_params:
                continue
            if clean(v) != clean(other.key_params[k]):
                differences.append([k,v,other.key_params[k]])
                #print(f"Key: {k:<35} Self: {v:<25} Other: {other.key_params[k]}")
        for k, v in self.engine_params.items():
            if k not in other.engine_params:
                continue
            if clean(v) != clean(other.engine_params[k]):
                differences.append([k,v,other.engine_params[k]])
                #print(f"Key: {k:<35} Self: {v:<25} Other: {other.engine_params[k]}")
        for k, v in self.other_params.items():
            if k not in other.other_params:
                continue
            if clean(v) != clean(other.other_params[k]):
                differences.append([k,v,other.other_params[k]])
                #print(f"Key: {k:<35} Self: {v:<25} Other: {other.other_params[k]}")

        #print("Extra Parameters in Other:")
        for k, v in other.key_params.items():
            if k == "solFilesDirectory" or k == "dataDirectory":
                continue
            if k not in self.key_params:
                missing.append([k, other.key_params[k]])
                #print(f"Key: {k:<35} Self: None   Other: {other.key_params[k]}")
        for k, v in other.engine_params.items():
            if k not in self.engine_params:
                missing.append([k, other.engine_params[k]])
                #print(f"Key: {k:<35} Self: None   Other: {other.engine_params[k]}")
        for k, v in other.other_params.items():
            if k not in self.other_params:
                missing.append([k, other.other_params[k]])
                #print(f"Key: {k:<35} Self: None   Other: {other.other_params[k]}")
        return differences, missing

    def diff(self, other):
        differences, missing = self.get_diffs(other)
        for k, ours, theirs in differences:
            print(f"Key: {k:<35} Self: {ours:<25} Other: {theirs}")

        if len(missing) > 0:
            print("Extra Parameters in Other:")
            for k, theirs in missing:
                print(f"Key: {k:<35} Self: None   Other: {theirs}")

if __name__ == "__main__":
    default  = CaseInpFile()
    modified = CaseInpFile()

    modified.set_param("MachNumber", 1.4)
    modified.set_param("alpha", 4.3)


        
