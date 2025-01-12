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


def train_contrastive_loss(datasetPath = 'SignDataset', numEpochs = 25, batch_size=10, hidden_size=32, lstm_layers=1, z_size=3, num_negative=120, temp=0.9):
    print(f'    Temp: {temp}    output_size: {z_size}    numEpochs: {numEpochs}')
    classes = ['j', 'z_pinky', 'Wait', 'fire','Bye','Hellow', 'Yes','L', 'z_index', 'Thumbs up']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.LSTMModel_No_Image_Critique(batch_size, hidden_size, output_size=z_size, numLayers=lstm_layers)
    model = model.to(device)
    model.train()

    dataset = data_set.ContrastiveLearningDataset(datasetPath, negative_size=num_negative)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_ft = optim.Adam(model.parameters(), lr = 0.0005, weight_decay=1e-8)

    lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=5, gamma = 0.5)

    lossPerBatch = []
    avgLossPerEpoch = [0 for i in range(numEpochs)]
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    for epoch in range(numEpochs):
        print(f'    Epoch {epoch+1}/{numEpochs}')
        model.train()
        epoch_loss = 0
        num_batches = 0
        for anchor, anchor_image, positive_sample, positive_image, negative_samples, negative_images, _ in dataloader:
            anchor = anchor.to(torch.float32).to(device)
            positive_sample = positive_sample.to(torch.float32).to(device)
            negative_samples = negative_samples.to(torch.float32).to(device)
            optimizer_ft.zero_grad()
            with(torch.set_grad_enabled(True)):
                anchor_output = model(anchor)
                positive_output = model(positive_sample)
                negative_output = model(negative_samples[:,0,:,:])

                numerator = torch.exp(F.cosine_similarity(anchor_output, positive_output)/temp)
                denom = torch.exp(F.cosine_similarity(anchor_output, negative_output)/temp)
                for i in range(1, num_negative):
                    negative_output = model(negative_samples[:,i,:,:])
                    denom = denom + torch.exp(F.cosine_similarity(anchor_output, negative_output)/temp)
                #denom = denom/num_negative
                #print(f"Numerator: {numerator}")
                #print(f"Denom: {denom}")
                loss = (-torch.log(numerator/denom)).mean()
                #print(f"Loss: {loss.item()}")
                lossPerBatch.append(loss.item())
                epoch_loss = epoch_loss + loss.item()
                num_batches += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer_ft.step()
        avgLossPerEpoch[epoch] = epoch_loss / num_batches
        print(f'    Avg Loss: {avgLossPerEpoch[epoch]}')
        print('-'*20)
        lr.step()
        #print(f'Training Loss: {avgLossPerEpoch[epoch]}')
    #print(f'Training Duration: {time.time() - start_tim)
    training_time = time.time()-start_time
    return model, lossPerBatch, avgLossPerEpoch, training_time

def train_classifier(repModel,  z_size, train_dir='Validation', test_dir = 'Test', numEpochs=50, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = data_set.ClassificationLearningDataset(train_dir, model = repModel, device = device)
    test_dataset = data_set.ClassificationLearningDataset(test_dir, model = repModel, device = device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = models.NNModel(input_size = z_size, output_size=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-8)

    lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=7, gamma = 0.5)
    
    lossPerBatch = []
    avgLossPerEpoch = [0 for i in range(numEpochs)]
    train_accuracy = [0 for i in range(numEpochs)]
    test_accuracy = [0 for i in range(numEpochs)]
    start_time = time.time()
    for epoch in range(numEpochs):
        if((epoch+1)%10 == 0):
            print(f'    Epoch {epoch+1}/{numEpochs}')
        correct = 0
        total = 0
        model.train()
        epoch_loss = 0
        num_batches = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()
            with(torch.set_grad_enabled(True)):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                lossPerBatch.append(loss.item())
                epoch_loss = epoch_loss + loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        avgLossPerEpoch[epoch] = epoch_loss/num_batches
        train_accuracy[epoch] = 100*(correct/total)
        test_correct = 0
        test_total = 0
        model.eval()
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with(torch.no_grad()):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted==labels).sum().item()
        test_accuracy[epoch] = 100*(test_correct/test_total)
        if((epoch+1)%10 == 0):
            print(f'    Avg Loss: {avgLossPerEpoch[epoch]}    Train Acc: {train_accuracy[epoch]}    Test Acc: {test_accuracy[epoch]}')
            print('-'*20)
        lr.step()
    training_time = time.time()-start_time
    return model, lossPerBatch, avgLossPerEpoch, train_accuracy, test_accuracy, training_time








def train_AutoEncoder(z_size =2, train_dir='SignDataset', numEpochs=50, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = data_set.ContrastiveLearningDataset(train_dir, negative_size=1)
    #test_dataset = data_set.ClassificationLearningDataset(test_dir, model = repModel, device = device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    encoder = models.LSTMModel_No_Image_Critique(batch_size, 32, output_size=z_size, numLayers=1).to(device)
    decoder = models.LSTMModel_AutoEncode().to(device)
    criterion = nn.MSELoss()

    optimizer_ft = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=7, gamma = 0.5)
    lossPerBatch = []
    avgLossPerEpoch = [0 for i in range(numEpochs)]
    start_time = time.time()
    for epoch in range(numEpochs):
        if((epoch+1)%10 == 0):
            print(f'    Epoch {epoch+1}/{numEpochs}')
        encoder.train()
        decoder.train()
        epoch_loss = 0
        num_batches = 0
        for inputs, _, _, _, _, _ , _ in train_dataloader:
            inputs = inputs.to(torch.float32).to(device)
            optimizer_ft.zero_grad()
            with(torch.set_grad_enabled(True)):
                zspace = encoder(inputs)
                outputs = decoder(zspace)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer_ft.step()

                lossPerBatch.append(loss.item())
                epoch_loss = epoch_loss + loss.item()
                num_batches += 1
        avgLossPerEpoch[epoch] = epoch_loss/num_batches
        
        if((epoch+1)%10 == 0):
            print(f'    Avg Loss: {avgLossPerEpoch[epoch]}')
            print('-'*20)
        lr.step()
    training_time = time.time()-start_time
    torch.save(encoder.state_dict(), 'encoder')
    torch.save(decoder.state_dict(), 'decoder')




'''
model, batchLoss, avgLoss, training_time = train_contrastive_loss(numEpochs = 0)
print(batchLoss)
print(avgLoss)
print(training_time)
model = torch.load('GoodModel_10_class_25_epoch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = data_set.ClassificationLearningDataset("Test", model = model, device = device)
assert 1 == 0
dataset = data_set.ContrastiveLearningDataset("Test", negative_size=1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

classZ = {}
fireClass = []
model.eval()


cmap = get_cmap(10)
print("Generating Visualization")
time_start = time.time()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r','g','b','c','m']
shape = ['o', '^']
for anchor, anchor_image, positive_sample, positive_image, negative_samples, negative_images, classID in dataloader:
    anchor = anchor.to(torch.float32).to(device)
    anchor_output = model(anchor).detach().cpu().squeeze().numpy()
    if(classID.item() in classZ.keys()):
        classZ[classID.item()].append(anchor_output)
    else:
        classZ[classID.item()] = []
        classZ[classID.item()].append(anchor_output)
for i, k in enumerate(sorted(classZ.keys())):
    x1 = [coord[0] for coord in classZ[k]]
    y1 = [coord[1] for coord in classZ[k]]
    z1 = [coord[2] for coord in classZ[k]]
    ax.scatter(x1, y1, z1, color=colors[i%5], marker=shape[int(i/5)], label=str(k))  # Blue points
plt.legend()
plt.show()

assert 1==0


'''
'''
pyplot.figure()
pyplot.plot(percentageTrain)
pyplot.title('Training Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.show(block = False)

pyplot.figure()
pyplot.plot(percentageVal)
pyplot.title('Validation Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.show(block = False)

pyplot.figure()
pyplot.plot(trainLoss)
pyplot.title('Training Loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.show(block = False)

pyplot.figure()
pyplot.plot(valLoss)
pyplot.title('Validation Loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.show(block = False)
'''
'''
torch.save(model.state_dict(), 'lstm_continuous_training_NIGHT_BEFORE_SKIP_2_FINAL')
#model.load_state_dict(torch.load('lstm_final_training'))

#####RUNNING PART#####
    
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


np_arr = np.zeros(shape = (1024, 21, 3), dtype = float)
#torch.save(model.state_dict(), 'lstn_30_epoch_3_2_24')
waitingTIme = input("Press Enter to Begin...")

# For webcam input:
torch.set_grad_enabled(False)
cap = cv2.VideoCapture(0)
printing = True
index = 0
model.eval()
handTime = 0
totalTime = 0
numMeasures = 0
hPrev = torch.zeros((2, hidden_size)).to(device)
cPrev = torch.zeros((2, hidden_size)).to(device)
totalInput = torch.zeros((30, 21*3)).to(device)
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
           
            #for hand_landmarks in results.multi_hand_landmarks:
            #        mp_drawing.draw_landmarks(
            #        image,
            #        hand_landmarks,
            #        mp_hands.HAND_CONNECTIONS,
            #        mp_drawing_styles.get_default_hand_landmarks_style(),
            #        mp_drawing_styles.get_default_hand_connections_style())
            totalInput[0:29, :] = totalInput[1:30, :].clone()
            totalInput[-1, :] = inputTensor
            #results, (hPrev, cPrev) = model.predict(inputTensor.to(device).unsqueeze(0), hPrev, cPrev)
            results = model(totalInput)[-1, :]
            results = torch.nn.functional.softmax(results) 
            _, preds = torch.max(results.unsqueeze(0),1)
            predicted = letters[preds]
            prev60[0:59] = prev60[1:60] 
            prev60[59] = preds

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
            '''
'''
            if(preds == 0):
                #print('Scissors')
                predicted = 'A'
            elif(preds == 1):
                #print('Paper')
                predicted = 'B'
            elif(preds == 2):
                #print('Paper')
                predicted = 'C'
            elif(preds == 3):
                #print("Rock")
                predicted = 'D'
            elif(preds == 4):
                #print("Rock")
                predicted = 'J'
            else:
                predicted = 'RPS'
            '''
'''
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        t2 = time.time()
        totalTime = ((totalTime*(timer-1))+(t2-t1))/timer
        print(f'Avg Hand Detection Time: {handTime} Secs')
        print(f'Avg Total Time: {totalTime} Secs')
        print('--------------------------------------')
        cv2.putText(image, predicted, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, 'Currently Learning: ' + letters[currLearning], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
'''

if __name__ == '__main__':
    train_AutoEncoder()
