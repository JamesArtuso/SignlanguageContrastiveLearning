import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import get_data
import time
import models
from matplotlib import pyplot


batch_size = 1
hidden_size = 256
lstm_layers = 2
output_size = 26
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
allData = get_data.getDataMotionSequence()
valData = get_data.getDataMotionSequence('Test', size=10)
print(allData.shape)

model = models.OLD_LSTMModel(batch_size, hidden_size, output_size=10, numLayers=lstm_layers)
model.to(device)


#model.load_state_dict(torch.load('lstm_continuous_training'))

model.train()

#model.load_state_dict(torch.load('lstm_continuous_training_NIGHT_BEFORE_SKIP_2_CHECKPOINT_3'))

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model.parameters(), lr = 0.00125/2, weight_decay=1e-8)
#optimizer_ft = optim.SGD(model.parameters(), lr = 0.005, momentum=0.9)

lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=14000*5*2, gamma = 0.5)



numEpochs = 80

percentageTrain = []
percentageVal = []
trainLoss = []
valLoss = []
start_time = time.time()
for epoch in range(numEpochs):
    numCorrect = 0
    numTotal = 0
    valCorrect = 0
    valTotal = 0
    print(f'Epoch {epoch}/{numEpochs-1}')
    print('-'*10)
    model.train()
    runningLoss = 0
    for inputs, labels in allData:
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer_ft.zero_grad()

        with(torch.set_grad_enabled(True)):
            outputs = model(inputs)
            _, preds = torch.max(outputs.squeeze(0), 1)
            loss = criterion(outputs.squeeze(0)[-1], labels.squeeze(1)[-1])
            loss.backward()
            optimizer_ft.step()
            lr.step()
        numTotal += 1
        numCorrect += preds[-1] == labels.squeeze(1)[-1].item()#torch.sum(preds == labels.squeeze(1).data)
        runningLoss += loss.item()
    trainLoss.append(runningLoss)
    print(f'Train Number correct: {numCorrect}/{numTotal}')
    print(f'Train Accuracy: {100*(numCorrect/numTotal)}%')
    print(f'Training Loss: {runningLoss}')
    model.eval()
    runningLoss = 0
    for inputs, labels in valData:
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with(torch.no_grad()):
            outputs = model(inputs)
            _, preds = torch.max(outputs.squeeze(0), 1)
            loss = criterion(outputs.squeeze(0)[-1], labels.squeeze(1)[-1])
        valTotal += 1
        valCorrect += preds[-1] == labels.squeeze(1)[-1].item()
        runningLoss += loss.item()
    valLoss.append(runningLoss)
    print(f'Val Number correct: {valCorrect}/{valTotal}')
    print(f'Val Accuracy: {100*(valCorrect/valTotal)}%')
    print(f'Val Loss: {runningLoss}')
    percentageTrain.append(float(numCorrect/numTotal))
    percentageVal.append(float(valCorrect/valTotal))

    if(epoch % 10 == 0 and epoch != 0):
        torch.save(model.state_dict(), 'lstm_continuous_training_NIGHT_BEFORE_SKIP_2_CHECKPOINT_' + str(int(epoch/10)+3))
print(f'Training Time: {time.time() - start_time}')

pyplot.figure()
pyplot.plot(percentageTrain)
pyplot.title('Training Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.savefig("OldTrainAcc.png")

pyplot.figure()
pyplot.plot(percentageVal)
pyplot.title('Validation Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.savefig("OldTestAcc.png")

pyplot.figure()
pyplot.plot(trainLoss)
pyplot.title('Training Loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.savefig("OldTrainLoss.png")

pyplot.figure()
pyplot.plot(valLoss)
pyplot.title('Validation Loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.savefig("OldTestLoss.png")

torch.save(model.state_dict(), 'lstm_old')
#model.load_state_dict(torch.load('lstm_final_training'))
assert 1 == 0
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