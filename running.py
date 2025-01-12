#####RUNNING PART#####
    
import cv2
import mediapipe as mp
import numpy as np


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
import pickle

dataset = data_set.EvaluationDataset('Test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print('Done Importing')
classes = ['j', 'z_pinky', 'Wait', 'fire','Bye','Hello', 'Yes','L', 'z_index', 'Thumbs up']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Setting up MediaPipe')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
t = 0.5
z = 2

print("Setting up models")
contrast_model = models.LSTMModel_No_Image_Critique(10, 32, output_size=z, numLayers=1)
contrast_model.load_state_dict(torch.load(f"Output2/contrastModel_t_{t}_z{z}", weights_only=True))
contrast_model = contrast_model.to(device)
contrast_model.eval()

class_model = models.NNModel(input_size = z, output_size=10)
class_model.load_state_dict(torch.load(f"Output2/classModel_t_{t}_z{z}", weights_only=True))
class_model = class_model.to(device)
class_model.eval()

#Set up pyplot
with open('saved_dictonary.pkl', 'rb') as f:
    classZ = pickle.load(f)


colors = ['r','g','b','c','m']
shape = ['o', '^']

fig, ax1 = plt.subplots()
z = 2
for i, k in enumerate(sorted(classZ.keys())):
    x1 = [coord[0] for coord in classZ[k]]
    y1 = [coord[1] for coord in classZ[k]]
    ax1.scatter(x1, y1, color = colors[i % 5], marker=shape[int(i/5)], label = classes[i])
handles, labels = plt.gca().get_legend_handles_labels()
sorted_handles_labels = sorted(zip(labels, handles))
sorted_labels, sorted_handles = zip(*sorted_handles_labels)
plt.legend(sorted_handles, sorted_labels, loc='lower left')
plt.axis('equal')
plt.xlabel(r"$z_{\text{1}}$")
plt.ylabel(r"$z_{\text{2}}$")
plt.title(r"Contrastive Compression $z_{\text{dim}} = 2$")  
live_point, = ax1.plot([],[],'ro')

def update_point(x, y):
    x_new = x 
    y_new = y
    live_point.set_data([x_new], [y_new])  # Update the live point data

    plt.draw()  # Redraw the figure
    plt.pause(0.01)  # Pause to make the update visible

np_arr = np.zeros(shape = (1024, 21, 3), dtype = float)
#torch.save(model.state_dict(), 'lstn_30_epoch_3_2_24')
waitingTIme = input("Press Enter to Begin...")

# For webcam input:
torch.set_grad_enabled(False)
cap = cv2.VideoCapture(0)
printing = True
index = 0
handTime = 0
totalTime = 0
numMeasures = 0
totalInput = torch.zeros((60, 21*3)).to(device)
timer = 0
prev60 = [-1 for i in range(60)]
currLearning = 0

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.05) as hands:
    while cap.isOpened():
        timer += 1
        t1 = time.time()
        numMeasures += 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        t2 = time.time()
        handTime = ((handTime*(timer-1))+(t2-t1))/timer
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        predicted = 'No Hand'
        if results.multi_hand_landmarks:
            inputTensor = torch.zeros(21*3)
            for hand_landmarks in results.multi_hand_world_landmarks:
                for i , _landmark in enumerate(hand_landmarks.landmark):
                    inputTensor[3*i] = _landmark.x
                    inputTensor[3*i+1] = _landmark.y
                    inputTensor[3*i+2] = _landmark.z
           

            totalInput[0:59, :] = totalInput[1:60, :].clone()
            totalInput[-1, :] = inputTensor
            zspace = contrast_model(totalInput.unsqueeze(0))
            zspaceCPU = zspace.cpu()
            update_point(zspaceCPU[0,0], zspaceCPU[0,1])
            results = class_model(zspace)
            _, preds = torch.max(results, 1)
            predicted = classes[preds]

            #Curr Learning
            numFull = 0
            fullLearn = True
            for i in range(59, -1, -1):
                if(prev60[i] != currLearning):
                    fullLearn = False
                    break
                else:
                    numFull += 1
            if(numFull > 5):
                cv2.rectangle(image, (50, 10), (int(50+500*numFull/60), 20), (255,255,255))
            if(fullLearn):
                currLearning += 1
            #for i in range(output_size):
            #    cv2.rectangle(image, (50, 100 + 16*i), (int(50+100*results[i].item()), 108 + 16*i), (255,255,255))
            predicted = classes[preds]

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        t2 = time.time()
        totalTime = ((totalTime*(timer-1))+(t2-t1))/timer
        print(f'Avg Hand Detection Time: {handTime} Secs')
        print(f'Avg Total Time: {totalTime} Secs')
        print('--------------------------------------')
        cv2.putText(image, predicted, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, 'Currently Learning: ' + classes[currLearning], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()

