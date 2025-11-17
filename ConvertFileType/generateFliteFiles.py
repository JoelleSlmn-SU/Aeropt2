import os
import numpy as np


def writeCoordsFile(xCoords, yCoords, filename, filepath, gen):
    coords = []
    for i in range(len(xCoords)):
        coords.append(str(xCoords[i])+','+str(yCoords[i])+'\n')
    
    directory = "Generation " + str(gen)
    path = os.path.join(filepath, directory)
    if not os.path.exists(path):
        os.mkdir(path)

    csvpath = path
    gmt = open(f'{csvpath}\\{filename}', 'w')
    gmt.writelines(coords)
    gmt.close()
    
def generateBoundData(x1, y1, x2, y2, FFx, FFy, filepath):
    temp1 = [i for i in range(1,len(x1)+1)] + [len(x1) + j for j in range(1,len(x2)+1)] + [len(x1)+len(x2) + i for i in range(1,len(FFx)+1)]
    temp2 = [i + 1 for i in range(1,len(y1))] + [1] + [len(y1) + j for j in range(2,len(y2)+1)] + [len(y1)+1] + [len(y1)+len(y2) + 1 + i for i in range(1,len(FFy))] + [len(y1)+len(y2) + 1]
    temp3 = [1 for i in range(len(x1)+len(x2))] + [3 for i in range(len(FFx))]

    bound_data = []
    for i in range(np.size(temp1)):
        bound_data.append(str(temp1[i])+','+str(temp2[i])+','+str(temp3[i])+'\n')

    csvname = "boundData.csv"
    gmt = open(f'{filepath}\\{csvname}', 'w')
    gmt.writelines(bound_data)
    gmt.close() 