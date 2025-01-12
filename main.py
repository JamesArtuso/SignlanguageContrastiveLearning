import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import data_set
import time
import models
import matplotlib.pyplot as plt
import training
import pickle
import os

def plot_items(arrays, testParams, name, xlabel, ylabel, saveDir):
    plt.figure()
    for i, arr in enumerate(arrays):
        plt.plot(arr, label=str(testParams[i]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.savefig(saveDir + '\\'+name.replace(' ', '_')+".png")

def plot_bar(arr, testParams, name, xlabel, ylabel, saveDir):
    plt.figure()
    labels = [str(i) for i in testParams]
    plt.bar(labels, arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(saveDir + '\\'+name.replace(' ', '_')+".png")


def main():
    outputDir = 'Output2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testTemps = [0.2, 0.3, 0.5, 0.7, 0.8]
    testZdim = [2, 3, 5, 8]
    testParams = [(t, z) for z in testZdim for t in testTemps]
    contrastiveEpochAvgLosses = []
    contrastiveBatchLosses = []
    contrastiveTrainingTimes = []
    contrastiveNumParams = []
    classEpochAvgLosses = []
    classBatchLosses = []
    classTrainAccs = []
    classTestAccs = []
    classTrainTimes = []
    classNumParams = []
    bestTestAcc = 0
    bestIdx = 0
    bestContrastModel = None
    bestClassModel = None
    for i, (t, z) in enumerate(testParams):
        print(f"----------Testing Temp-Zdim pair: ({t}, {z})----------\n")
        print("*****Training Contrastive Loss*****")
        contrastModel, batchLoss, avgLoss, training_time = training.train_contrastive_loss(datasetPath = 'SignDataset', numEpochs = 25, batch_size=10, hidden_size=32, lstm_layers=1, z_size=z, num_negative=100, temp=t)
        torch.save(contrastModel.state_dict(), outputDir+'\\contrastModel_t_'+str(t)+'_z'+str(z))
        contrastiveNumParams.append(sum(p.numel() for p in contrastModel.parameters() if p.requires_grad))
        contrastiveEpochAvgLosses.append(avgLoss)
        contrastiveBatchLosses.append(batchLoss)
        contrastiveTrainingTimes.append(training_time)
        classModel, lossPerBatch, avgLossPerEpoch, train_accuracy, test_accuracy, class_training_time = training.train_classifier(contrastModel, z_size=z)
        classNumParams.append(sum(p.numel() for p in classModel.parameters() if p.requires_grad))
        torch.save(classModel.state_dict(), outputDir+'\\classModel_t_'+str(t)+'_z'+str(z))
        classEpochAvgLosses.append(avgLossPerEpoch)
        classBatchLosses.append(lossPerBatch)
        classTrainAccs.append(train_accuracy)
        classTestAccs.append(test_accuracy)
        classTrainTimes.append(class_training_time)
        if(test_accuracy[-1] > bestTestAcc):
            bestTestAcc = test_accuracy[-1]
            bestIdx = i
            bestContrastModel=contrastModel
            bestClassModel = classModel

    plot_items(contrastiveEpochAvgLosses, testParams, 'Contrastive Loss Per Epoch', 'Epoch', 'Loss', outputDir)
    plot_items(contrastiveBatchLosses, testParams, 'Contrastive Loss Per Batch', 'Batch', 'Loss', outputDir)
    plot_items(classEpochAvgLosses, testParams, 'Classification Loss Per Epoch', 'Epoch', 'Loss', outputDir)
    plot_items(classBatchLosses, testParams,'Classification Loss Per Batch', 'Batch', 'Loss', outputDir)
    plot_items(classTrainAccs, testParams, 'Training Accuracy', 'Epoch', 'Accuracy', outputDir)
    plot_items(classTestAccs, testParams, 'Testing Accuracy', 'Epoch', 'Accuracy', outputDir)
    plot_bar(contrastiveTrainingTimes, testParams, 'Contrastive Training Time', 'Temp-Z Dimension Pair', 'Time (sec)', outputDir)
    plot_bar(classTrainTimes, testParams, 'Classification Training Time', 'Temp-Z Dimension Pair', 'Time (sec)', outputDir)
    plot_bar(contrastiveNumParams, testParams, 'Contrastive Model Size', 'Temp-Z Dimension Pair', 'Number of Parameters', outputDir)
    plot_bar(classNumParams, testParams, 'Classification Model Size', 'Temp-Z Dimension Pair', 'Number of Parameters', outputDir)
    with open(outputDir+'\\trainData'+str(testParams[i])+'.pkl', 'wb') as f:
        saveArr = [contrastiveEpochAvgLosses,
                    contrastiveBatchLosses,
                    classEpochAvgLosses,
                    classBatchLosses,
                    classTrainAccs,
                    classTestAccs,
                    contrastiveTrainingTimes,
                    classTrainTimes,
                    contrastiveNumParams,
                    classNumParams,
                    testParams]
        pickle.dump(saveArr, f)
    print(bestIdx)
    print(testParams[bestIdx])


if __name__ == "__main__":
    main()