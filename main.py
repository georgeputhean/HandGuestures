import cv2
import mediapipe as mp

def count_fingers(hand_landmarks):
    # Assuming the hand is approximately horizontal, we can use the positions of specific landmarks to count fingers.
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # Assuming that the hand is facing the camera, we can use the y-coordinate of the landmarks to count fingers.
    fingers = 0
    if thumb_tip.y > index_tip.y:
        fingers += 1
    if index_tip.y > middle_tip.y:
        fingers += 1
    if middle_tip.y > ring_tip.y:
        fingers += 1
    if ring_tip.y > pinky_tip.y:
        fingers += 1

    return fingers

def detect_hand_gestures():

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the number of fingers by counting based on hand landmarks
                num_fingers = count_fingers(hand_landmarks)

                # Draw the hand landmarks and the number of fingers on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, f'Fingers: {num_fingers}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gestures', image)

        if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand_gestures()
