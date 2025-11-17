import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from IPython.display import display
import random
import os
import sys
from saveFiles import saveDesignVar
sys.path.append(os.path.dirname('ShapeParameterization'))
sys.path.append(os.path.dirname('FileRW'))
sys.path.append(os.path.dirname('Flite'))
from ShapeParameterization.IntakeParaCN import *
from ShapeParameterization.controlNodes import *
from FileRW.loadDatFile import *
from Flite.runFlite import *
from Flite.intakeAIP import *

cwd = os.getcwd()

def initPop(controlNodes, disp, numAgents, gen):
    for i in range(1, numAgents+1):
        controlNodesInit = [[0,0] for i in range(len(controlNodes))]
        for j in range(len(controlNodes)):
            controlNodesInit[j][0] = controlNodes[j][0] + random.uniform(disp[0][0], disp[0][1])
            controlNodesInit[j][1] = controlNodes[j][1] + random.uniform(disp[1][0], disp[1][1])
        
        saveDesignVar(controlNodesInit, gen, i)
    
    
def getFitness(gen, numAgent):
    fitness = random.randrange(10)
    
    #drag = run_matlab(cwd, gen, numAgent)
    #PR = getPressureRecovery()
    
    return fitness


def EAOptimisation(fitnessValues, parameterBounds, gen, numAgents, SR=0.6, MR=0.3):
    # Load Control Nodes
    controlNodes = []
    for agent in range(1, numAgents+1):
        controlNodesAgent = loadDesignVar(gen-1, agent)
        controlNodesAgent = [[float(n) for n in e] for e in controlNodesAgent]
        controlNodes.append(controlNodesAgent)
        
    # Select parents based on fitness - Tournament Selection
    numParents = int(numAgents * SR)
    parentIndices = np.argsort(fitnessValues)[:numParents]

    # Creating a new population by mutating and recombining parents
    newPopulation = []
    for _ in range(numParents):
        parent1 = controlNodes[np.random.choice(parentIndices)]
        parent2 = controlNodes[np.random.choice(parentIndices)]

        numNodes = len(parent1)
        crossoverPoint = np.random.randint(1, numNodes - 1)
        child = np.vstack((parent1[:crossoverPoint], parent2[crossoverPoint:]))

        for i in range(len(child)):
            if np.random.rand() < MR:
                if i < len(child):
                    # Adjust x and y coordinates within parameterBounds by a random offset
                    x_offset = np.random.uniform(parameterBounds[0][0], parameterBounds[0][1])
                    y_offset = np.random.uniform(parameterBounds[1][0], parameterBounds[1][1])
                    child[i][0] = child[i][0] + x_offset
                    child[i][1] = child[i][1] + y_offset
                else:
                #    x_offset = np.random.uniform(-0.005, 0.005)
                    x_offset = 0
                #    y_offset = np.random.uniform(-0.005, 0.005)
                    y_offset = 0
                    child[i][0] = child[i][0] + x_offset
                    child[i][1] = child[i][1] + y_offset
        child = child.tolist()
        
        newPopulation.append(child)

    for i in range(len(parentIndices)):
        controlNodes[parentIndices[i]] = newPopulation[i]

    for agent in range(1, len(controlNodes)+1):
        saveDesignVar(controlNodes[agent-1], gen, agent)


def runEA(controlNodes, parameterBounds, numGenerations, numAgents, nc, nr):    
    # Configuration Parameters
    mutationRate = 0.1
    selectionRatio = 0.6

    # Create plot for fitness Values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fitnessValues = []
    for generation in range(1, numGenerations+1):
        if generation == 1:
            initPop(controlNodes, parameterBounds, numAgents, generation)
        else:
            EAOptimisation(fitnessValuesGen, parameterBounds, generation, numAgents, selectionRatio, mutationRate)
        
        fitnessValuesGen = []    
        for agent in range(1, numAgents+1):
            runParameterization(generation, agent, nc, nr)
            fitnessValuesGen.append(getFitness(generation, agent))
            
        fitnessValues.append(fitnessValuesGen)
        
    ax.set_xlim(0, numGenerations)
    plt.scatter([generation]*len(fitnessValuesGen), fitnessValuesGen, marker='.', color="black")
    plt.title("Fitness Values")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    display(fig)
    plt.pause(10)
        
            

def main():    
    controlNodes, nCowl, nRamp = selectControlNodes()
    saveDesignVar(controlNodes, 0, 1)
    # Define the allowed displacement for x and y displacement for each point
    parameterBounds = [[-0.05,0.05], [-0.05,0.05]]
    
    numGens = 2
    numAgents = 5
    runEA(controlNodes, parameterBounds, numGens, numAgents, nCowl, nRamp)


# Run the main function
if __name__ == "__main__":
    main()