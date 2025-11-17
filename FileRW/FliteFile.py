import os

class FliteFile():
    def __init__(self, ext="", exc_p=[], exc_f=[]):
        self.ext   = ext
        self.exc_p = exc_p
        self.exc_f = exc_f
        
    @staticmethod
    def getFileExt(ext, exc_p=[], exc_f=[]):
        """
            Get the first file that has file extension ".ext"
            Excluding all of *exc* for exc in exc_p
            Excluding all of exc for exc in exc_f
        """
        for file in os.listdir():
            valid = False
            if file.endswith(f".{ext}"):
                valid = True
            for ex in exc_p:
                if ex in file :
                    valid = False
            for ex in exc_f:
                if file == ex:
                    valid = False
            if valid:
                return file
        return None
        
    @staticmethod
    def getFileExtOptions(ext, exc_p=[], exc_f=[]):
        """
            Allow the user to select from any file that ends in *.ext
            Excluding all of *exc* for exc in exc_p
            Excluding all of exc for exc in exc_f
        """
        options = []
        for file in os.listdir():
            valid = False
            if file.endswith(f".{ext}"):
                valid = True
            for ex in exc_p:
                if ex in file :
                    valid = False
            for ex in exc_f:
                if file == ex:
                    valid = False
            if valid:
                options.append(file)
        return FliteFile.selectFromOptions(options)

    @staticmethod
    def getMatch(part):
        """
            Get the first file that contains *part*
        """
        for file in os.listdir():
            if part in file:
                return file
        return None

    @staticmethod
    def getMatchOptions(part):
        """
            Allow the user to select from any file that contains *part*
        """
        options = []
        for file in os.listdir():
            if part in file:
                options.append(file)
        return FliteFile.selectFromOptions(options)
    
    @staticmethod
    def selectFromOptions(options):
        if len(options) == 0:
            return None
        if len(options) == 1:
            return options[0]
        req = "Multiple possibilities found. Please select from the following:\n"
        for i,o in enumerate(options):
            req += f"{str(i): >.3} = {o}\n"
        res = -1
        poss = [x for x in range(len(options))]
        while int(res) not in poss:
            print(f"res={res}, poss={poss}")
            res = input(req)
        return options[int(res)]

    def findFile(self):
        ext   = self.ext
        exc_p = self.exc_p
        exc_f = self.exc_f
        for file in os.listdir():
            valid = False
            if file.endswith(f".{ext}"):
                valid = True
            for ex in exc_p:
                if ex in file :
                    valid = False
            for ex in exc_f:
                if file == ex:
                    valid = False
            if valid:
                return file
        print("nothing found...!")
        return 
    

if __name__ == "__main__":
    filename = FliteFile.getFirstMatch("batchfile")
    #ff = FliteFile(ext="plt", exc_p=["_v1","_v2"], exc_f=["base.plt"])
    #ff = FliteFile(ext="bco", exc_p=["_v1","_v2"], exc_f=["base.plt"])
    #filename = ff.findFile()
    print(filename)