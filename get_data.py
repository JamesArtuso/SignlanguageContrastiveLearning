import numpy as np
from torch import tensor
from torch import float32 as fl32
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

rotations = np.array([[[0, -1, 0],[1, 0, 0], [0,0,1]],[[0, 1, 0],[-1, 0, 0], [0,0,1]], [[0, 0, 1], [0,1,0], [-1,0,0]],[[0, 0, -1], [0,1,0], [1,0,0]], [[1,0,0], [0,0,-1],[0,1,0]],[[1,0,0], [0,0,1],[0,-1,0]]])


def getData():
    scissorsData = np.load('RPSNodeDataset\\scissorsLandmarks.npy')
    paperData = np.load('RPSNodeDataset\\paperLandmarks.npy')
    rockData = np.load('RPSNodeDataset\\rockLandmarks.npy')

    allData = np.zeros(shape = (scissorsData.shape[0]+ paperData.shape[0]+rockData.shape[0], 2), dtype = object)
    for i in range(scissorsData.shape[0]):
        allData[i, 0] = tensor((scissorsData[i, : , : ].T).T, dtype = fl32)
        allData[i, 1] = tensor([0])

    for i in range(paperData.shape[0]):
        allData[i + scissorsData.shape[0], 0] = tensor((paperData[i, : , : ].T).T, dtype = fl32)
        allData[i + scissorsData.shape[0], 1] = tensor([1])

    for i in range(rockData.shape[0]):
        allData[i + scissorsData.shape[0] + paperData.shape[0], 0] = tensor((rockData[i, : , : ].T).T, dtype = fl32)
        allData[i + scissorsData.shape[0] + paperData.shape[0], 1] = tensor([2])

    np.random.shuffle(allData)
    return allData

def getDataFlat():
    scissorsData = np.load('RPSNodeDataset\\scissorsFlatLandmarks.npy')
    paperData = np.load('RPSNodeDataset\\paperFlatLandmarks.npy')
    rockData = np.load('RPSNodeDataset\\rockFlatLandmarks.npy')

    allData = np.zeros(shape = (scissorsData.shape[0]+ paperData.shape[0]+rockData.shape[0], 2), dtype = object)
    for i in range(scissorsData.shape[0]):
        allData[i, 0] = tensor(scissorsData[i, :], dtype = fl32)
        allData[i, 1] = tensor([0])

    for i in range(paperData.shape[0]):
        allData[i + scissorsData.shape[0], 0] = tensor(paperData[i, :], dtype = fl32)
        allData[i + scissorsData.shape[0], 1] = tensor([1])

    for i in range(rockData.shape[0]):
        allData[i + scissorsData.shape[0] + paperData.shape[0], 0] = tensor(rockData[i, : ], dtype = fl32)
        allData[i + scissorsData.shape[0] + paperData.shape[0], 1] = tensor([2])

    np.random.shuffle(allData)
    return allData



def getDataFlatSequence():
    scissorsData = np.load('RPSNodeDataset\\scissorsFlatLandmarks.npy')
    paperData = np.load('RPSNodeDataset\\paperFlatLandmarks.npy')
    rockData = np.load('RPSNodeDataset\\rockFlatLandmarks.npy')

    allData = np.zeros(shape = (scissorsData.shape[0]+ paperData.shape[0]+rockData.shape[0], 2), dtype = object)
    for i in range(scissorsData.shape[0]):
        allData[i, 0] = tensor(scissorsData[i, :], dtype = fl32).unsqueeze(0).repeat(30, 1)
        allData[i, 1] = tensor([0]).unsqueeze(0).repeat(30, 1)

    for i in range(paperData.shape[0]):
        allData[i + scissorsData.shape[0], 0] = tensor(paperData[i, :], dtype = fl32).unsqueeze(0).repeat(30, 1)
        allData[i + scissorsData.shape[0], 1] = tensor([1]).unsqueeze(0).repeat(30, 1)

    for i in range(rockData.shape[0]):
        allData[i + scissorsData.shape[0] + paperData.shape[0], 0] = tensor(rockData[i, : ], dtype = fl32).unsqueeze(0).repeat(30, 1)
        allData[i + scissorsData.shape[0] + paperData.shape[0], 1] = tensor([2]).unsqueeze(0).repeat(30, 1)

    np.random.shuffle(allData)
    return allData


def getDataFlatSequenceExtended():
    scissorsData = np.load('RPSNodeDataset\\scissorsFlatLandmarksFinal.npy')
    paperData = np.load('RPSNodeDataset\\paperFlatLandmarksFinal.npy')
    rockData = np.load('RPSNodeDataset\\rockFlatLandmarksFinal.npy')

    allData = np.zeros(shape = (7*(scissorsData.shape[0]+ paperData.shape[0]+rockData.shape[0]), 2), dtype = object)
    currIndex = 0
    for i in range(scissorsData.shape[0]):
        allData[7*currIndex, 0] = tensor(scissorsData[i, :], dtype = fl32)
        allData[7*currIndex, 1] = tensor([0])
        handRotations = getRotation(np.expand_dims(scissorsData[i, :], axis=0), ranges = 2*np.pi)
        for k in range(handRotations.shape[0]):
            allData[7*currIndex + k+1, 0 ] = tensor(handRotations[k, :], dtype = fl32).squeeze(0)
            allData[7*currIndex + k+1, 1] = tensor([0])
        currIndex += 1

    for i in range(paperData.shape[0]):
        allData[7*currIndex, 0] = tensor(paperData[i, :], dtype = fl32)
        allData[7*currIndex, 1] = tensor([1])
        handRotations = getRotation(np.expand_dims(paperData[i, :], axis =0), ranges = 2*np.pi)
        for k in range(handRotations.shape[0]):
            allData[7*currIndex + k+1, 0 ] = tensor(handRotations[k, :], dtype = fl32).squeeze(0)
            allData[7*currIndex + k+1, 1] = tensor([1])
        currIndex += 1

    for i in range(rockData.shape[0]):
        allData[7*currIndex, 0] = tensor(rockData[i, :], dtype = fl32)
        allData[7*currIndex, 1] = tensor([2])
        handRotations = getRotation(np.expand_dims(rockData[i, :], axis = 0), ranges = 2*np.pi)
        for k in range(handRotations.shape[0]):
            allData[7*currIndex + k + 1, 0 ] = tensor(handRotations[k, :], dtype = fl32).squeeze(0)
            allData[7*currIndex + k + 1, 1] = tensor([2])
        currIndex += 1

    np.random.shuffle(allData)
    return allData
'''
def getExtendedDataFlat():
    scissorsData = np.load('RPSNodeDataset\\scissorsFlatLandmarksFinal.npy')
    paperData = np.load('RPSNodeDataset\\paperFlatLandmarksFinal.npy')
    rockData = np.load('RPSNodeDataset\\rockFlatLandmarksFinal.npy')

    allData = np.zeros(shape = (3*(scissorsData.shape[0]+ paperData.shape[0]+rockData.shape[0], 2)), dtype = object)
    for i in range(scissorsData.shape[0]):
        allData[3*i, 0] = tensor(scissorsData[i, :], dtype = fl32)
        allData[3*i, 1] = tensor([0])
        for j in range(2):
            allData[3*i + 1 + j, 0] = tensor(scissorsData[i, :] @ getRandomRotationMatrix(), dtype = fl32)
            allData[3*i, 1] = tensor([0])

    for i in range(paperData.shape[0]):
        allData[3*i + 3*scissorsData.shape[0], 0] = tensor(paperData[i, :], dtype = fl32)
        allData[3*i + 3*scissorsData.shape[0], 1] = tensor([1])
        for j in range(2):
            allData[3*i + 1 + j + 3*scissorsData.shape[0], 0] = tensor(paperData[i, :] @ getRandomRotationMatrix(), dtype = fl32)
            allData[3*i + 1 + j + 3*scissorsData.shape[0], 1] = tensor([1])


    for i in range(rockData.shape[0]):
        allData[3*i + 3*scissorsData.shape[0] + 3*paperData.shape[0], 0] = tensor(rockData[i, : ], dtype = fl32)
        allData[3*i + 3*scissorsData.shape[0] + 3*paperData.shape[0], 1] = tensor([2])
        for j in range(2):
            allData[3*i + 1 + j + 3*scissorsData.shape[0] + 3*paperData.shape[0], 0] = tensor(rockData[i, : ] @ getRandomRotationMatrix(), dtype = fl32)


    np.random.shuffle(allData)
    return allData
'''

def getDataMotionSequence(file = 'SignDataset', size = 20):
    onlyfiles = [f for f in listdir(file) if isfile(join(file, f))]
    '''
    AData = np.load('SignDataset\\A.npy')
    BData = np.load('SignDataset\\B.npy')
    CData = np.load('SignDataset\\C.npy')
    DData = np.load('SignDataset\\D.npy')
    JData = np.load('SignDataset\\J.npy')
    RPSData = np.load('SignDataset\\RPS.npy')
    '''
    #allData = np.zeros(shape = (20*len(onlyfiles), 2), dtype = object)
    allData = np.zeros(shape = (size*10, 2), dtype = object)
    currIndex = 0
    for i in range(len(onlyfiles)):
        currData = np.load(file+'\\'+onlyfiles[i], allow_pickle=True)
        for j in range(currData[0].shape[0]):
            #if(i == 3):
            #    animateSequence(currData[j, :])
            #    k = input('1012030')
            allData[currIndex, 0] = tensor(currData[0][j, :], dtype = fl32)
            allData[currIndex, 1] = tensor([i]).unsqueeze(0).repeat(currData[0].shape[1], 1)
            currIndex += 1
    '''
    for i in range(AData.shape[0]):
        allData[i, 0] = tensor(AData[i, :], dtype = fl32)
        allData[i, 1] = tensor([0]).unsqueeze(0).repeat(30, 1)

    for i in range(BData.shape[0]):
        allData[i + AData.shape[0], 0] = tensor(BData[i, :], dtype = fl32)
        allData[i + AData.shape[0], 1] = tensor([1]).unsqueeze(0).repeat(30, 1)

    for i in range(CData.shape[0]):
        allData[i + AData.shape[0] + BData.shape[0], 0] = tensor(CData[i, :], dtype = fl32)
        allData[i + AData.shape[0] + BData.shape[0], 1] = tensor([2]).unsqueeze(0).repeat(30, 1)

    for i in range(DData.shape[0]):
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0], 0] = tensor(DData[i, :], dtype = fl32)
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0], 1] = tensor([3]).unsqueeze(0).repeat(30, 1)
        

    for i in range(JData.shape[0]):
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0] + DData.shape[0], 0] = tensor(JData[i, :], dtype = fl32)
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0] + DData.shape[0], 1] = tensor([4]).unsqueeze(0).repeat(30, 1)

    for i in range(RPSData.shape[0]):
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0] + DData.shape[0] + JData.shape[0], 0] = tensor(RPSData[i, :], dtype = fl32)
        allData[i + AData.shape[0] + BData.shape[0] + CData.shape[0] + DData.shape[0] + JData.shape[0], 1] = tensor([5]).unsqueeze(0).repeat(30, 1)
    #for i in range(rockData.shape[0]):
    #    allData[i + scissorsData.shape[0] + paperData.shape[0], 0] = tensor(rockData[i, : ], dtype = fl32).unsqueeze(0).repeat(30, 1)
    #    allData[i + scissorsData.shape[0] + paperData.shape[0], 1] = tensor([2]).unsqueeze(0).repeat(30, 1)
    '''
    np.random.shuffle(allData)
    return allData


def getDataMotionSequenceExtended():
    onlyfiles = [f for f in listdir('SignDataset') if isfile(join('SignDataset', f))]
    '''
    AData = np.load('SignDataset\\A.npy')
    BData = np.load('SignDataset\\B.npy')
    CData = np.load('SignDataset\\C.npy')
    DData = np.load('SignDataset\\D.npy')
    JData = np.load('SignDataset\\J.npy')
    RPSData = np.load('SignDataset\\RPS.npy')
    '''
    #allData = np.zeros(shape = (20*len(onlyfiles), 2), dtype = object)
    allData = np.zeros(shape = (520*7, 2), dtype = object)
    currIndex = 0
    for i in range(len(onlyfiles)):
        currData = np.load('SignDataset\\'+onlyfiles[i])
        for j in range(currData.shape[0]):
            allData[7*currIndex, 0] = tensor(currData[j, :], dtype = fl32)
            allData[7*currIndex, 1] = tensor([i]).unsqueeze(0).repeat(currData.shape[1], 1)
            currHand = currData[j, :]
            handRotations = getRotation(currHand)
            for k in range(handRotations.shape[0]):
                allData[7*currIndex + k+1, 0] = tensor(handRotations[k,:], dtype = fl32)
                allData[7*currIndex + k+1, 1] = tensor([i]).unsqueeze(0).repeat(currData.shape[1], 1)
            currIndex += 1
    np.random.shuffle(allData)
    return allData

def getRotation(handSequence, ranges = 2*(np.pi/3)):
    avgX = 0
    avgY = 0
    avgZ = 0
    handSequenceReshaped = np.zeros((handSequence.shape[0], 21, 3))
    for i in range(6):
        t1 = np.random.rand()*ranges - ranges/2
        t2 = np.random.rand()*ranges - ranges/2
        t3 = np.random.rand()*ranges - ranges/2
        r1 = np.array([[1,0,0], [0, np.cos(t1), -np.sin(t1)], [0, np.sin(t1), np.cos(t1)]])
        r2 = np.array([[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]])
        r3 = np.array([[np.cos(t3), -np.sin(t3), 0], [np.sin(t3), np.cos(t3), 0], [0, 0, 1]])
        rotation = r1 @ r2 @ r3
        rotations[i] = rotation
    for i in range(handSequence.shape[0]):
        for j in range(21):
            handSequenceReshaped[i, j, 0] = handSequence[i, 3*j]
            avgX += handSequence[i, 3*j]
            handSequenceReshaped[i, j, 1] = handSequence[i, 3*j+1]
            avgY += handSequence[i, 3*j+1]
            handSequenceReshaped[i, j, 2] = handSequence[i, 3*j+2]
            avgZ += handSequence[i, 3*j+2]
    handSequenceRotated = np.zeros((6, handSequenceReshaped.shape[0], 63))
    avgX /= handSequence.shape[0]*21
    avgY /= handSequence.shape[0]*21
    avgZ /= handSequence.shape[0]*21
    for i in range(6):
        for j in range(handSequenceReshaped.shape[0]):
            handSequenceRotated[i, j, :] = (np.matmul(rotations[i], np.subtract(handSequenceReshaped[j, :,:], np.array([avgX, avgY, avgZ]).T).T).T + np.array([avgX, avgY, avgZ]).T).flatten()
    return handSequenceRotated

'''
def getRandomRotation(handSequence):
    t1 = np.random.rand()*np.pi*2
    t2 = np.random.rand()*np.pi*2
    t3 = np.random.rand()*np.pi*2
    r1 = [[1,0,0], [0, np.cos(t1), -np.sin(t1)], [0, np.sin(t1), np.cos(t1)]]
    r2 = [[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]]
    r3 = [[np.cos(t3), -np.sin(t3), 0], [np.sin(t3), np.cos(t3), 0], [0, 0, 1]]
    rotation = r1 @ r2 @ r3

    avgX = 0
    avgY = 0
    avgZ = 0
    handSequenceReshaped = np.zeros((handSequence.shape[0], 21, 3))
    for i in range(handSequence.shape[0]):
        for j in range(21):
            handSequenceReshaped[i, j, 0] = handSequence[i, 3*j]
            avgX += handSequence[i, 3*j]
            handSequenceReshaped[i, j, 1] = handSequence[i, 3*j+1]
            avgY += handSequence[i, 3*j+1]
            handSequenceReshaped[i, j, 2] = handSequence[i, 3*j+2]
            avgZ += handSequence[i, 3*j+2]
    handSequenceRotated = np.zeros((6, handSequenceReshaped.shape[0], 63))
    avgX /= handSequence.shape[0]*21
    avgY /= handSequence.shape[0]*21
    avgZ /= handSequence.shape[0]*21
    for i in range(6):
        for j in range(handSequenceReshaped.shape[0]):
            handSequenceRotated[i, j, :] = (np.matmul(rotations[i], np.subtract(handSequenceReshaped[j, :,:], np.array([avgX, avgY, avgZ]).T).T).T + np.array([avgX, avgY, avgZ]).T).flatten()
    return handSequenceRotated
    return rotation
'''

def animateSequence(handSequence):
    handSequenceReshaped = np.zeros((handSequence.shape[0], 21, 3))
    for i in range(handSequence.shape[0]):
        for j in range(21):
            handSequenceReshaped[i, j, 0] = handSequence[i, 3*j]
            handSequenceReshaped[i, j, 1] = handSequence[i, 3*j+1]
            handSequenceReshaped[i, j, 2] = handSequence[i, 3*j+2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show(block = False)
    index = 0
    handsLines = [(0,1), (1,2), (2,3), (3,4), (0, 17), (0, 5), (5,9), (9,13), (13, 17), (5, 6), (6,7), (7,8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
    while True:
        ax.clear()
        ax.axis('off')
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.scatter(handSequenceReshaped[index, :, 0],handSequenceReshaped[index, :, 1], handSequenceReshaped[index, :, 2], color = 'b')
        for i in range(len(handsLines)):
            ax.plot(handSequenceReshaped[index, handsLines[i], 0],handSequenceReshaped[index, handsLines[i], 1], handSequenceReshaped[index, handsLines[i], 2], color = 'b')
        index = (index + 1) % handSequenceReshaped.shape[0]

        fig.canvas.draw() 
        fig.canvas.flush_events() 


#getDataFlatSequenceExtended()
#getDataMotionSequenceExtended()