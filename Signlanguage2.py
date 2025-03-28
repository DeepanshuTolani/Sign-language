import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  # For text-to-speech (optional)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Enhanced gesture dictionary with more signs
gesture_dict = {
    "open_palm": ("Hello", 0.9),    # (word, base confidence)
    "pointing": ("You", 0.85),
    "fist": ("No", 0.9),

    "peace": ("Yes", 0.8),          # Index and middle fingers up
    "thumb_up": ("Good", 0.85),     # Thumb up, others down
    "l_shape": ("I Love YOU ", 0.75)       # Index and thumb extended
}

# Initialize text-to-speech engine (optional)
try:
    tts_engine = pyttsx3.init()
    tts_available = True
except Exception:
    tts_available = False
    print("Text-to-speech unavailable.")

def analyze_hand_landmarks(landmarks):
    """
    Analyze hand landmarks to determine gesture with confidence.
    Returns: (gesture_name, confidence_score)
    """
    wrist_y = landmarks.landmark[0].y
    thumb_tip_y = landmarks.landmark[4].y
    index_tip_y = landmarks.landmark[8].y
    middle_tip_y = landmarks.landmark[12].y
    ring_tip_y = landmarks.landmark[16].y
    pinky_tip_y = landmarks.landmark[20].y

    # Check finger states (up if tip y-coordinate < wrist y-coordinate)
    fingers_up = [tip_y < wrist_y for tip_y in [thumb_tip_y, index_tip_y, middle_tip_y, ring_tip_y, pinky_tip_y]]

    # Gesture detection logic
    if all(fingers_up[1:]):  # All fingers except thumb up
        return ("open_palm", 0.95)
    elif fingers_up[1] and not any(fingers_up[2:]):  # Only index up
        return ("pointing", 0.9)
    elif not any(fingers_up[1:]):  # No fingers up (fist)
        return ("fist", 0.95)
    elif fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:  # Peace sign
        return ("peace", 0.85)
    elif fingers_up[0] and not any(fingers_up[1:]):  # Thumb up
        return ("thumb_up", 0.9)
    elif fingers_up[1] and fingers_up[0] and not any(fingers_up[2:]):  # L shape
        return ("l_shape", 0.8)
    return ("unknown", 0.0)

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Show hand gestures to the camera. Press 'q' to quit.")
    
    # Variables for sentence building
    sentence = []
    last_gesture = None
    gesture_stable_time = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        current_gesture = "unknown"
        confidence = 0.0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks with color based on detection
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green for landmarks
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Blue for connections
                )

                # Analyze gesture
                current_gesture, confidence = analyze_hand_landmarks(hand_landmarks)

                # Stabilize gesture detection (require gesture to hold for 1 second)
                current_time = time.time()
                if current_gesture == last_gesture and current_gesture != "unknown":
                    if current_time - last_time >= 1.0:  # Gesture held for 1 second
                        if current_gesture in gesture_dict and gesture_stable_time == 0:
                            word = gesture_dict[current_gesture][0]
                            sentence.append(word)
                            if tts_available:
                                tts_engine.say(word)
                                tts_engine.runAndWait()
                            gesture_stable_time = current_time
                else:
                    last_gesture = current_gesture
                    last_time = current_time
                    gesture_stable_time = 0

        # UI Overlay
        # Background rectangle for text
        cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)  # Gray background
        # Current gesture
        gesture_text = f"Gesture: {gesture_dict.get(current_gesture, ('Unknown', 0))[0]} ({confidence:.2f})"
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Sentence display
        sentence_text = "Sentence: " + " ".join(sentence[-5:])  # Show last 5 words
        cv2.putText(frame, sentence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Impressive Sign Language Translator", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if tts_available:
        tts_engine.stop()

if __name__ == "__main__":
    main()