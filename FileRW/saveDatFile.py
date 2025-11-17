import numpy as np
import csv
import os

def saveSelectedControlNodes(controlNodes, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, "optimisationCN.txt")

    data = controlNodes
    
    np.savetxt(filename, data, fmt=['%.3f','%.3f','%.3f'])
    
    return

def saveControlPoints(controlPoints, output_dir):
    directory = os.path.join(output_dir, "Control Nodes")
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "surfaceControlPoints.txt")

    # Open in append mode
    with open(filename, 'a') as file:
        for point in controlPoints:
            line = " ".join(f"{coord:.6f}" for coord in point)  # Format to 6 decimal places
            file.write(line + "\n")
            
    return