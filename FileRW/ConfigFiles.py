import os

def config_folder():
    file_path   = os.path.abspath(__file__)
    fileio_path = os.path.dirname(file_path)
    aeropt_path = os.path.dirname(fileio_path)
    python_path = os.path.dirname(aeropt_path)
    cfg = os.path.join(python_path, "ConfigFiles")
    return cfg + os.sep