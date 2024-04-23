import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

def main():
    st.set_page_config(page_title="Sign Language Recognition", page_icon=":diamond:")
    st.title("Sign Language Recognition")

    # Load the Random Forest Classifier model
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'A',10:'B',11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',18:'J',19:'K',20:'L',21:'M',22:'N',23:'O',24:'P',25:'Q',26:'R',27:'S',28:'T',29:'U',30:'V',31:'W',32:'X',33:'Y',34:'Z',35:'hello',36:'afternoon',37:'good',38:'morning',39:'evening',40:'how are you',41:'good day',42:'happy birthday',43:'happy anniversary',44:'red',45:'blue',46:'green',47:'yellow',48:'white',49:'black',50:'pink',51:'brown',52:'gold',54:'orange',55:'please',56:'youre welcome',57:'excuse me',58:'im sorry',59:'nice to meet you'}
    
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame from the camera.")
            break

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

                # Pad the feature vector to have 100 features
                while len(data_aux) < 100:
                    data_aux.append(0.0)

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
