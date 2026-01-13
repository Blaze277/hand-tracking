import cv2
import mediapipe as mp
import time

prev = time.time()
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style()
            )
            
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
            wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
            cv2.putText(frame, f"{label} ({score:.2f})",
                        (wrist_x - 40, wrist_y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    now = time.time()
    fps = 1 / (now - prev) if (now - prev) > 0 else 0
    prev = now
    
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

hands.close()
#say you want this
