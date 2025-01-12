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
import os
import get_data
import seaborn as sns
import pickle

def show_z_dim_backup(t, z, auto=False):
    assert z <= 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    encode_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    contrast_model.load_state_dict(torch.load(f"Output2/contrastModel_t_{t}_z{z}", weights_only=True))
    encode_model.load_state_dict(torch.load(f"encoder", weights_only=True))
    #contrast_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    contrast_model = contrast_model.to(device).eval()
    encode_model = encode_model.to(device).eval()
    dataset = data_set.EvaluationDataset('Test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    classes = ['J', 'Pinky Z', 'Wait', 'Fire','Bye','Hello', 'Yes','L', 'Index Z', 'Thumbs up']
    classZ = {}
    encodeZ = {}
    fireClass = []
    contrast_model.eval()

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax1 = axes[0]

    print("Generating Visualization")
    if(z == 3):
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')

    colors = ['r','g','b','c','m']
    shape = ['o', '^']
    for input, classID in dataloader:
        input = input.to(torch.float32).to(device)
        anchor_output = contrast_model(input).detach().cpu().squeeze().numpy()
        encode_output = encode_model(input).detach().cpu().squeeze().numpy()
        if(classID.item() in classZ.keys()):
            classZ[classID.item()].append(anchor_output)
            encodeZ[classID.item()].append(encode_output)
        else:
            classZ[classID.item()] = []
            classZ[classID.item()].append(anchor_output)
            encodeZ[classID.item()] = []
            encodeZ[classID.item()].append(encode_output)
    for i, k in enumerate(sorted(classZ.keys())):
        x1 = [coord[0] for coord in classZ[k]]
        y1 = [coord[1] for coord in classZ[k]]
        if z == 3:
            z1 = [coord[2] for coord in classZ[k]]
            ax1.scatter(x1, y1, z1, color=colors[i%5], marker=shape[int(i/5)], label=classes[i])  # Blue points
        else:
            ax1.scatter(x1, y1, color=colors[i%5], marker=shape[int(i/5)], label=classes[i])  # Blue points
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(labels, handles))
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    plt.legend(sorted_handles, sorted_labels)
    plt.axis('equal')
    plt.xlabel(r"$z_{\text{1}}$")
    plt.ylabel(r"$z_{\text{2}}$")
    plt.title(r"Contrastive Compression $z_{\text{dim}} = 2$")
    if(auto):
        plt.title(r"Autoencoder Compression $z_{\text{dim}} = 2$")
    plt.show()

def show_z_dim(t, z, auto=False):
    assert z <= 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    encode_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    contrast_model.load_state_dict(torch.load(f"Output2/contrastModel_t_{t}_z{z}", weights_only=True))
    encode_model.load_state_dict(torch.load(f"encoder", weights_only=True))
    #contrast_model = models.LSTMModel_No_Image_Critique(1, 32, output_size=z, numLayers=1)
    contrast_model = contrast_model.to(device).eval()
    encode_model = encode_model.to(device).eval()
    dataset = data_set.EvaluationDataset('Test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    classes = ['J', 'Pinky Z', 'Wait', 'Fire','Bye','Hello', 'Yes','L', 'Index Z', 'Thumbs up']
    classZ = {}
    encodeZ = {}
    contrast_model.eval()

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax1 = axes[0]

    print("Generating Visualization")
    if(z == 3):
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')

    colors = ['r','g','b','c','m']
    shape = ['o', '^']
    for input, classID in dataloader:
        input = input.to(torch.float32).to(device)
        anchor_output = contrast_model(input).detach().cpu().squeeze().numpy()
        encode_output = encode_model(input).detach().cpu().squeeze().numpy()
        if(classID.item() in classZ.keys()):
            classZ[classID.item()].append(anchor_output)
            encodeZ[classID.item()].append(encode_output)
        else:
            classZ[classID.item()] = []
            classZ[classID.item()].append(anchor_output)
            encodeZ[classID.item()] = []
            encodeZ[classID.item()].append(encode_output)
    with open('saved_dictonary.pkl', 'wb') as f:
        pickle.dump(classZ, f)
    for i, k in enumerate(sorted(classZ.keys())):
        x1 = [coord[0] for coord in classZ[k]]
        y1 = [coord[1] for coord in classZ[k]]
        if z == 3:
            z1 = [coord[2] for coord in classZ[k]]
            ax1.scatter(x1, y1, z1, color=colors[i%5], marker=shape[int(i/5)], label=classes[i])  # Blue points
        else:
            ax1.scatter(x1, y1, color=colors[i%5], marker=shape[int(i/5)], label=classes[i])  # Blue points
    ax1.set_xlabel(r"$z_{\text{1}}$")
    ax1.set_ylabel(r"$z_{\text{2}}$")
    ax1.set_title(r"Contrastive Compression $z_{\text{dim}} = $" + str(z))
    ax1.axis('equal')
    ax1.legend(loc='best')
    
    # Plot for the encoder model
    ax2 = axes[1]
    if z == 3:
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')  # 3D plot for z_dim=3
    for i, k in enumerate(sorted(encodeZ.keys())):
        x2 = [coord[0] for coord in encodeZ[k]]
        y2 = [coord[1] for coord in encodeZ[k]]
        if z == 3:
            z2 = [coord[2] for coord in encodeZ[k]]
            ax2.scatter(x2, y2, z2, color=colors[i % 5], marker=shape[int(i / 5)], label=classes[i])
        else:
            ax2.scatter(x2, y2, color=colors[i % 5], marker=shape[int(i / 5)], label=classes[i])

    # Set labels and title for the second plot
    ax2.set_xlabel(r"$z_{\text{1}}$")
    ax2.set_ylabel(r"$z_{\text{2}}$")
    ax2.set_title(r"Autoencoder Compression $z_{\text{dim}} = $" + str(z))
    ax2.axis('equal')
    ax2.legend(loc='best')

    # Display the plots
    plt.tight_layout()
    plt.show()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=10 
    hidden_size=32 
    lstm_layers=1
    t = 0.5 #Best Output1
    z = 8 #Best Output1
    t = 0.3 #Best Output2
    z = 8 #Best Output2
    classes = ['J', 'Pinky Z', 'Wait', 'Fire','Bye','Hello', 'Yes','L', 'Index Z', 'Thumbs up']

    contrast_model = models.LSTMModel_No_Image_Critique(batch_size, hidden_size, output_size=z, numLayers=lstm_layers)
    contrast_model.load_state_dict(torch.load(f"Output2/contrastModel_t_{t}_z{z}", weights_only=True))
    contrast_model = contrast_model.to(device)
    contrast_model.eval()

    class_model = models.NNModel(input_size = z, output_size=10)
    class_model.load_state_dict(torch.load(f"Output2/classModel_t_{t}_z{z}", weights_only=True))
    class_model = class_model.to(device)
    class_model.eval()
    old_model = models.OLD_LSTMModel(1, 256, output_size=10, numLayers=2)
    old_model.load_state_dict(torch.load(f"lstm_old", weights_only=True))
    old_model = old_model.to(device)
    old_model.eval()


    oldNumParams = sum(p.numel() for p in old_model.parameters() if p.requires_grad)
    classNumParams = sum(p.numel() for p in class_model.parameters() if p.requires_grad) + sum(p.numel() for p in contrast_model.parameters() if p.requires_grad)

    dataset = data_set.EvaluationDataset('Test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    confMat = np.zeros((10,10),dtype=int)
    confMatOld = np.zeros((10, 10), dtype=int)
    start_time = time.time()
    avg_time = 0
    old_time = 0
    correct = 0
    oldCorrect = 0
    total = 0
    for input, label in dataloader:
        input = input.to(torch.float32).to(device)
        label = label.to(device)
        start_time = time.time()
        output = contrast_model(input)
        output = class_model(output)
        avg_time += time.time()-start_time
        start_time= time.time()
        oldOut = old_model(input)
        oldOut = oldOut[:,-1,:]
        old_time += time.time()-start_time
        _, predicted = torch.max(output, 1)
        _, oldPred = torch.max(oldOut, 1)
        total += 1
        correct += (predicted==label).sum().item()
        oldCorrect += (oldPred==label).sum().item()
        confMat[label.item()][predicted.item()] = confMat[label.item(), predicted.item()]+1
        confMatOld[label.item()][oldPred.item()] = confMatOld[label.item()][oldPred.item()]+1
    print(total)
    print("Confusion Matrix")
    print(confMat)
    plt.figure(figsize=(10,8))
    plt.title("Our Confusion Matrix")
    sns.heatmap(confMat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.savefig("ourConfMat.png")
    print(f'Accuracy: {correct/total}')
    print(f'Average Compute Time (sec): {avg_time/total}')
    print(f'Number of Parameters: {classNumParams}')
    print('\n\n')
    print("Old Confusion Matrix")
    print(confMatOld)
    plt.figure(figsize=(10,8))
    plt.title("Old Confusion Matrix")
    sns.heatmap(confMatOld, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.savefig("oldConfMat.png")
    print(f'Old Accuracy: {oldCorrect/total}')
    print(f'Average Compute Time (sec): {old_time/total}')
    print(f'Number of Parameters: {oldNumParams}')
    show_z_dim(0.5, 2)
    #seq, classId = dataset[97]
    #print(seq.shape)
    #print(classId)
    #get_data.animateSequence(seq)
    


if __name__ == "__main__":
    main()