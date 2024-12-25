import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # For the cooldown timer

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()
except KeyError:
    print("Error: Model file does not contain 'model' key.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)


labels_dict = {i: chr(65 + i) for i in range(26)}

detected_word = ""
previous_letter = None  
last_append_time = time.time()  
cooldown_seconds = 4 

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    cv2.putText(
        frame,
        'Made By: Omar Elzarka, Ahmed Elmehalawy, Mohamed Ghazal, Ahmed Fouad',
        (0, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

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

        x1 = max(0, int(min(x_) * W) - 10)
        y1 = max(0, int(min(y_) * H) - 10)
        x2 = min(W, int(max(x_) * W) + 10)
        y2 = min(H, int(max(y_) * H) + 10)

        try:
            data_aux = np.asarray(data_aux).reshape(1, -1)  # Reshape for prediction
            prediction = model.predict(data_aux)
            predicted_character = labels_dict[int(prediction[0])]

            
            current_time = time.time()
            if current_time - last_append_time > cooldown_seconds:
                detected_word += predicted_character
                last_append_time = current_time  
                previous_letter = predicted_character  
        except KeyError:
            predicted_character = "Unknown"
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_character = "B"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):  # Press 'a' to exit
        break
    elif key == ord(' '):  # Press spacebar to add a space
        detected_word += " "  # Append a space to the word
    elif key == 8:  # Press backspace to delete the last character
        detected_word = detected_word[:-1]  # Remove the last character

    # Display the detected word on the frame
    """cv2.putText(
        frame,
        f"Word: {detected_word}",
        (10, H - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,"""
    

    cv2.imshow('Intelligent Systems Department - Image Processing Project', frame)

cap.release()
cv2.destroyAllWindows()
