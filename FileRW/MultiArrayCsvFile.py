import sys
import numpy as np

class MultiArrayCsvFile:
    """
        filename should include desired extension
        
        prints data as k-len(v):\\ v    
    """
    def __init__(self, filename:str=""):
        self.filename = filename

    def read(self) -> dict:
        """read data in and return as a dict"""
        data_arrays = {}
        with open(self.filename, "r") as f:
            current_key = None
            lines_to_read = 0
            while True:
                if lines_to_read == 0:
                    if current_key is not None:
                        data_arrays[current_key] = np.array(data_arrays[current_key])
                    l = f.readline().strip().strip("\n")
                    if not l:
                        break
                    current_key, ltr = l.split("-")
                    lines_to_read = int(ltr[:-1])
                    data_arrays[current_key] = []
                else:
                    l = f.readline().strip().strip("\n")
                    data_arrays[current_key].append([float(x) for x in l.strip("[").strip("]").split()])
                    lines_to_read -= 1
        return data_arrays

    def write(self, data:dict) -> None:
        """write out a dictionary in similar fashion to numpy savez but in human readable csv"""
        with open(self.filename, "w") as f:
            for k,v in data.items():
                f.write(f"{k}-{len(v)}:\n")
                if len(v) > 0:
                    if type(v) == np.ndarray:
                        f.write(f"{np.array2string(v, max_line_width=350)}\n")
                    else:
                        f.write(f"{v}\n")

if __name__ == "__main__":
    mac = MultiArrayCsvFile("test.csv")
    data = {}
    data["X"] = np.array([[1,1,1], [2,2,2], [3,3,3]])
    data["lift"] = np.array([[1],[2],[3]])
    data["drag"] = np.array([[5],[6],[7]])
    mac.write(data)
    new_data = mac.read()
    print(new_data)

