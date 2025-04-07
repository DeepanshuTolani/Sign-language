import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import random
import math

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# Enhanced gesture dictionary with more gestures and colors
GESTURES = {
    "open_palm": {"text": "Hello", "confidence": 0.95, "color": (0, 255, 0), "emoji": "👋"},
    "pointing": {"text": "You", "confidence": 0.9, "color": (255, 0, 0), "emoji": "👉"},
    "fist": {"text": "No", "confidence": 0.95, "color": (0, 0, 255), "emoji": "✊"},
    "peace": {"text": "Yes", "confidence": 0.85, "color": (255, 255, 0), "emoji": "✌️"},
    "thumb_up": {"text": "Good", "confidence": 0.9, "color": (0, 255, 255), "emoji": "👍"},
    "l_shape": {"text": "I Love You", "confidence": 0.8, "color": (255, 0, 255), "emoji": "🤟"},
    "three_fingers": {"text": "Thank You", "confidence": 0.85, "color": (255, 128, 0), "emoji": "🙏"},
    "four_fingers": {"text": "Wait", "confidence": 0.8, "color": (128, 0, 255), "emoji": "✋"},
    "ok_sign": {"text": "OK", "confidence": 0.9, "color": (0, 128, 255), "emoji": "👌"},
    "rock_on": {"text": "Rock On", "confidence": 0.8, "color": (255, 0, 128), "emoji": "🤘"}
}

# Initialize text-to-speech engine
class TextToSpeech:
    def __init__(self):
        self.engine = None
        self.available = False
        self.initialize()
    
    def initialize(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[1].id)  # Female voice
            self.available = True
        except Exception as e:
            print(f"Text-to-speech unavailable: {e}")
    
    def speak(self, text):
        if self.available:
            self.engine.say(text)
            self.engine.runAndWait()
    
    def stop(self):
        if self.available:
            self.engine.stop()

tts = TextToSpeech()

# Particle system for visual effects
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = random.randint(2, 5)
        self.speed = random.uniform(1, 3)
        self.angle = random.uniform(0, 2 * math.pi)
        self.life = random.randint(20, 40)
        self.alpha = 255
    
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.life -= 1
        self.alpha = int(255 * (self.life / 40))
        return self.life > 0
    
    def draw(self, frame):
        overlay = frame.copy()
        cv2.circle(overlay, (int(self.x), int(self.y)), self.size, self.color, -1)
        cv2.addWeighted(overlay, self.alpha/255, frame, 1 - self.alpha/255, 0, frame)

class GestureRecognizer:
    def __init__(self):
        self.history = []
        self.history_size = 10
        self.stable_time_threshold = 1.0  # seconds
    
    def recognize(self, landmarks):
        if not landmarks:
            return "unknown", 0.0
        
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        # Calculate distances between finger tips and wrist
        fingers_up = [
            thumb_tip.y < wrist.y - 0.05,  # Thumb
            index_tip.y < wrist.y - 0.05,   # Index
            middle_tip.y < wrist.y - 0.05,  # Middle
            ring_tip.y < wrist.y - 0.05,    # Ring
            pinky_tip.y < wrist.y - 0.05    # Pinky
        ]
    

        # Check for specific gestures
        thumb_index_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        if thumb_index_dist < 0.05 and not any(fingers_up[2:]):
            return "ok_sign", 0.9
        elif all(fingers_up[1:]):
            return "open_palm", 0.95
        elif fingers_up[1] and not any(fingers_up[2:]):
            return "pointing", 0.9
        elif not any(fingers_up[1:]):
            return "fist", 0.95
        elif fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return "peace", 0.85
        elif fingers_up[0] and not any(fingers_up[1:]):
            return "thumb_up", 0.9
        elif fingers_up[1] and fingers_up[0] and not any(fingers_up[2:]):
            return "l_shape", 0.8
        elif fingers_up[1] and fingers_up[2] and fingers_up[3] and not fingers_up[4]:
            return "three_fingers", 0.85
        elif all(fingers_up[1:4]) and fingers_up[4]:
            return "four_fingers", 0.8
        elif fingers_up[0] and fingers_up[4] and not any(fingers_up[1:4]):
            return "rock_on", 0.8
        return "unknown", 0.0
    
    def update_history(self, gesture, confidence):
        self.history.append((gesture, confidence))
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_stable_gesture(self):
        if not self.history:
            return "unknown", 0.0
        
        # Get most frequent gesture in history
        gestures = [g[0] for g in self.history]
        stable_gesture = max(set(gestures), key=gestures.count)
        
        # Calculate average confidence for this gesture
        confidences = [g[1] for g in self.history if g[0] == stable_gesture]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return stable_gesture, avg_confidence

class VisualEffects:
    def __init__(self):
        self.particles = []
        self.max_particles = 200
    
    def add_particles(self, x, y, color, count=10):
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                self.particles.append(Particle(x, y, color))
    
    def update_particles(self):
        self.particles = [p for p in self.particles if p.update()]
    
    def draw_particles(self, frame):
        for particle in self.particles:
            particle.draw(frame)
    
    def clear_particles(self):
        self.particles = []

class UIOverlay:
    @staticmethod
    def draw_info_panel(frame, gesture, confidence, sentence):
        # Create overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw semi-transparent panels
        cv2.rectangle(overlay, (0, 0), (width, 90), (20, 20, 40), -1)
        cv2.rectangle(overlay, (0, height-70), (width, height), (20, 20, 40), -1)
        
        # Add transparency
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Display current gesture
        if gesture in GESTURES:
            gesture_info = GESTURES[gesture]
            color = gesture_info["color"]
            text = f"{gesture_info['emoji']} {gesture_info['text']} ({confidence*100:.0f}%)"
            cv2.putText(frame, text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No gesture detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display sentence
        sentence_text = " ".join(sentence[-8:])  # Show last 8 words
        text_size = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, sentence_text, (text_x, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add watermark
        cv2.putText(frame, "AI Sign Language Translator Mde by king Deepanshu Tolani", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    @staticmethod
    def draw_gesture_confirmation(frame, gesture):
        if gesture in GESTURES:
            color = GESTURES[gesture]["color"]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), color, -1)
            text = f"Detected: {GESTURES[gesture]['text']} {GESTURES[gesture]['emoji']}"
            cv2.putText(overlay, text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def show_animated_intro():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    start_time = time.time()
    while time.time() - start_time < 5:  # 5 second intro
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Create overlay with gradient
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (width, height), (50, 0, 100), -1)
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 150, 255), 5, cv2.LINE_AA)
        
        # Add pulsing title
        elapsed = time.time() - start_time
        scale = 1 + 0.1 * math.sin(elapsed * 3)
        text = "AI Sign Language Translator By king Deepanshu tolani"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, scale, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2 - 50
        cv2.putText(overlay, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, scale, (255, 255, 255), 3)
        
        # Add loading animation
        progress = min(elapsed / 5, 1.0)
        cv2.rectangle(overlay, (width//4, height//2 + 50), 
                     (int(width//4 + (width//2) * progress), height//2 + 70), 
                     (0, 255, 255), -1)
        cv2.rectangle(overlay, (width//4, height//2 + 50), (3*width//4, height//2 + 70), 
                     (255, 255, 255), 2)
        
        # Blend with camera feed
        alpha = 0.7
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        cv2.imshow("AI Sign Language Translator By king deepanshu tolani", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def show_ready_screen():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    text = "Show Your Hand Gesture"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    
    for i in range(5, 0, -1):
        frame_copy = frame.copy()
        cv2.putText(frame_copy, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame_copy, f"Starting in {i}...", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add glow effect
        blur = cv2.GaussianBlur(frame_copy, (0, 0), 10)
        frame_copy = cv2.addWeighted(frame_copy, 1, blur, 0.5, 0)
        
        cv2.imshow("AI Sign Language Translator by king deepanshu tolani", frame_copy)
        cv2.waitKey(1000)
    
    cv2.destroyAllWindows()

def draw_landmarks_with_style(image, landmarks, effects):
    # Custom drawing specs
    connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
    landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=4)
    
    # Draw hand landmarks
    mp_drawing.draw_landmarks(
        image, landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_spec, connection_spec)
    
    # Add particles at landmark points
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        
        if random.random() < 0.3:
            color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
            effects.add_particles(x, y, color)
        
        # Draw glowing circles
        cv2.circle(image, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

def main():
    # Show animated intro
    show_animated_intro()
    show_ready_screen()

    # Initialize components
    recognizer = GestureRecognizer()
    effects = VisualEffects()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a window with better properties
    cv2.namedWindow("AI Sign Language Translator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Sign Language Translator", 800, 600)

    sentence = []
    last_gesture = None
    gesture_stable_time = 0
    last_time = time.time()
    frame_count = 0
    result = None  # Initialize result variable

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process frame every 2 frames for better performance
        if frame_count % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

        current_gesture = "unknown"
        confidence = 0.0
        
        if result and result.multi_hand_landmarks:  # Check if result exists first
            for hand_landmarks in result.multi_hand_landmarks:
                draw_landmarks_with_style(frame, hand_landmarks, effects)
                
                # Recognize gesture
                current_gesture, confidence = recognizer.recognize(hand_landmarks)
                recognizer.update_history(current_gesture, confidence)
                
                # Get stable gesture from history
                stable_gesture, avg_confidence = recognizer.get_stable_gesture()
                
                current_time = time.time()
                if stable_gesture == last_gesture and stable_gesture != "unknown":
                    if current_time - last_time >= 1.0:  # 1 second delay
                        if stable_gesture in GESTURES and gesture_stable_time == 0:
                            word = GESTURES[stable_gesture]["text"]
                            sentence.append(word)
                            
                            # Add visual confirmation
                            UIOverlay.draw_gesture_confirmation(frame, stable_gesture)
                            
                            # Speak the word
                            tts.speak(word)
                            
                            # Add celebration particles
                            color = GESTURES[stable_gesture]["color"]
                            for _ in range(50):
                                x = random.randint(0, frame.shape[1])
                                y = random.randint(0, frame.shape[0])
                                effects.add_particles(x, y, color)
                            
                            gesture_stable_time = current_time
                else:
                    last_gesture = stable_gesture
                    last_time = current_time
                    gesture_stable_time = 0

        # Update and draw particles
        effects.update_particles()
        effects.draw_particles(frame)

        # Draw UI overlay
        UIOverlay.draw_info_panel(frame, current_gesture, confidence, sentence)

        cv2.imshow("AI Sign Language Translator by king Deepanshu Tolani ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    tts.stop()

if __name__ == "__main__":
    main()
