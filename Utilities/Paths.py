import os
from importlib.machinery import SourceFileLoader

def up_one_level(path):
    '''
        Implements 'cd ../' on a string representation of a path.
    '''
    retval = ""
    path_split = path.strip("/").split("/")[:-1]
    for folder in path_split:
        retval += f"/{folder}"
    retval += "/"
    return retval

def file_line_count(filename):
    return sum(1 for line in open(filename, "r"))
    
def concat_fn_to_fp(path, name, sep=None):
    """
        Concatanate a path and a name 
        irrespective of wether the filepath 
        has the final slash applied or not.
    """
    if sep is None:
        return f"{path.rstrip(os.sep)}{os.sep}{name}"
    else:
        return f"{path.rstrip(sep)}{sep}{name}"

def load_functions(filepath):
    """
        Dynamically load a python file.

        Access by using the getattr function. for example:
            funcs = load_functions("functions.py")
            func = getattr(funcs, function_name)
            func()
    
        Arguments
        ----------
            filepath : str
                the full path of the file with extension
    
        Returns
        ----------
            returns the loaded module

        modified from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path?page=2&tab=scoredesc#tab-top
    """
    filepath = os.path.normpath(filepath)
    FULL_PATH = os.path.normpath(os.path.realpath(filepath))

    return SourceFileLoader(FULL_PATH, FULL_PATH).load_module()
