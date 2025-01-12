import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

numSamples = 20
sampleSize = 60
cvWidth = int(640/2)
cvHeight = int(480/2)
np_arr = np.zeros(shape = (numSamples,sampleSize , 21 * 3), dtype = float)
np_image_arr = np.zeros((numSamples, sampleSize, cvHeight, cvWidth))

# For webcam input:
cap = cv2.VideoCapture(0)
printing = True
index = -1
frame = 0
recording = False
countDown = 120
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if(recording):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            np_image_arr[index, frame, :,:] = cv2.resize((image[:,:,0]+image[:,:,1]+image[:,:,2])/3, (0,0), fx=0.5, fy=0.5)
            
            for hand_landmarks in results.multi_hand_world_landmarks:
                for i , _landmark in enumerate(hand_landmarks.landmark):
                    np_arr[index, frame, 3*i] = _landmark.x
                    np_arr[index, frame, 3*i + 1] = _landmark.y
                    np_arr[index, frame, 3*i + 2] = _landmark.z
                frame += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if(frame == sampleSize):
           recording = False
           index += 1
           if(index == numSamples):
                save_array = np.ndarray((2,), object)
                save_array[1] = np_image_arr
                save_array[0] = np_arr
                np.save('Validation\\class_9.npy', save_array)
                break
    else:
        countDown -= 1
        cv2.putText(image, str(int(countDown/30)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        if(countDown == 0):
           recording = True
           countDown = 120
           frame = 0
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1)) 
        
    # Flip the image horizontally for a selfie-view display.
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()