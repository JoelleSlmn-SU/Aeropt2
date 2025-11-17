import os
import csv

cwd = os.getcwd()

def saveDesignVar(designVarInit, gen, i):
    '''Saves design variables for each iteration'''
    mainPath = cwd + r"\Outputs\Intakes\Intake Control Nodes"
    directory = f"Generation {gen}"
    pathDV = os.path.join(mainPath, directory)

    # Create the directory if it does not exist
    os.makedirs(pathDV, exist_ok=True)
    filename = f"intakeCN{i}.csv"
    fullPath = os.path.join(pathDV, filename)
    
    with open(fullPath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(designVarInit)
            
def loadDesignVar(gen, agent):
    '''Loads 2D xy coordinates of control points .csv file'''
    mainPath = cwd + r"\Outputs\Intakes\Intake Control Nodes"
    directory = f"Generation {gen}"
    pathDV = os.path.join(mainPath, directory)
    filename = f"intakeCN{agent}.csv"
    
    file = open(f'{pathDV}\\{filename}', "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    
    return data