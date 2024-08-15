import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tensorflow import keras


model = keras.models.load_model('hand_sign_model.h5')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)


labels_dict = {0:'අම්මා',1: 'තාත්තා' ,2: 'සුභ උදෑසනක්',3: 'සුබ රාත්රියක්',4: 'ස්තුතියි', 5:'සමාවෙන්න',6:' පෑන', 7:' ඔව්', 8: 'නැත',9:'මොකක්ද',10: 'ඔබේ', 11:'මගේ',12:'මම',  13: 'මල',   }
while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            

            x_ = []
            y_ = []

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10


            data_aux = np.array(data_aux)
            
            res = model.predict(np.expand_dims(data_aux, axis=0)) 
           
            predicted_character = labels_dict[np.argmax(res)]


            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            ## Use iskpotab.ttf to write sinhala.
            fontpath = "iskpotab.ttf"  
            font = ImageFont.truetype(fontpath, 32)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((50, 5),  predicted_character, font = font, fill = (255, 255, 255))
            frame = np.array(img_pil)
            

    cv2.imshow('Sri Lankan Sign Recognition', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
