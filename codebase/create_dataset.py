import os
import pickle
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

DATA_DIR = './data'

data = []
labels = []
max_keypoints = 21

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))


            if len(data_aux) < max_keypoints * 2:
            # Pad with zeros
                data_aux.extend([0.0] * (max_keypoints * 2 - len(data_aux)))
            elif len(data_aux) > max_keypoints * 2:
            # Truncate the list
                data_aux = data_aux[:max_keypoints * 2]

            data.append(data_aux)
            labels.append(dir_)



data = np.array(data)
labels = np.array(labels)

np.save('data.npy', np.array(data))
np.save('labels.npy', np.array(labels))

print(data.shape)
print(labels.shape)



# Save the data and labels as .pickle file
#with open('data.pickle', 'wb') as f:
 #   pickle.dump({'data': data, 'labels': labels}, f)
