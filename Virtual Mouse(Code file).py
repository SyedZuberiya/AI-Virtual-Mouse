import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the hands model
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Get screen width and height for mapping the hand position to screen coordinates
screen_width, screen_height = pyautogui.size()

# Initialize previous positions
last_y_pos = 0
is_dragging = False
initial_pos = None

# Check if all fingers are close together (fist gesture)
def is_fist(hand_landmarks):
    finger_tips = [hand_landmarks.landmark[4], hand_landmarks.landmark[8], hand_landmarks.landmark[12], hand_landmarks.landmark[16], hand_landmarks.landmark[20]]
    distance = 0
    for i in range(1, len(finger_tips)):
        distance += np.abs(finger_tips[i].x - finger_tips[i-1].x) + np.abs(finger_tips[i].y - finger_tips[i-1].y)
    
    # If distance between fingers is small, it's a fist
    return distance < 0.1  # Threshold distance to be adjusted

# Check for an open hand gesture (fingers spread out)
def is_open_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    ring_finger_tip = hand_landmarks.landmark[16]
    pinky_finger_tip = hand_landmarks.landmark[20]
    
    # Calculate the distance between thumb and index finger
    distance_thumb_index = np.abs(thumb_tip.x - index_finger_tip.x) + np.abs(thumb_tip.y - index_finger_tip.y)
    distance_index_middle = np.abs(index_finger_tip.x - middle_finger_tip.x) + np.abs(index_finger_tip.y - middle_finger_tip.y)

    # If the hand is open (large distance between fingers), simulate scrolling
    return distance_thumb_index > 0.2 and distance_index_middle > 0.2

# Check for double-click gesture
def is_double_click(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    
    # Check for a quick movement between the thumb and index finger (similar to tapping)
    distance = np.abs(index_finger_tip.x - thumb_tip.x) + np.abs(index_finger_tip.y - thumb_tip.y)
    if distance < 0.05:
        pyautogui.doubleClick()

# Function to simulate dragging (click and hold)
def is_dragging_gesture(hand_landmarks):
    global is_dragging, initial_pos
    index_finger_tip = hand_landmarks.landmark[8]
    
    # Check if the fist gesture is detected for drag
    if is_fist(hand_landmarks):
        if not is_dragging:
            initial_pos = (index_finger_tip.x, index_finger_tip.y)  # Record the starting position
            is_dragging = True
    else:
        if is_dragging:
            # Calculate the movement and drag
            current_pos = (index_finger_tip.x, index_finger_tip.y)
            if initial_pos:
                delta_x = (current_pos[0] - initial_pos[0]) * screen_width
                delta_y = (current_pos[1] - initial_pos[1]) * screen_height
                pyautogui.move(delta_x, delta_y)
            initial_pos = current_pos
            is_dragging = False

# Volume control based on hand's vertical position
def control_volume(hand_landmarks):
    global last_y_pos
    index_finger_tip = hand_landmarks.landmark[8]
    current_y_pos = index_finger_tip.y
    
    # Compare the current position with the last position for detecting upward/downward movement
    if current_y_pos - last_y_pos > 0.02:
        pyautogui.hotkey('volumeup')  # Increase volume
    elif last_y_pos - current_y_pos > 0.02:
        pyautogui.hotkey('volumedown')  # Decrease volume
    
    last_y_pos = current_y_pos

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a later mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Loop through all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]

            # Map the coordinates from 3D space to 2D screen space
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move the mouse pointer to the location of the index finger tip
            pyautogui.moveTo(x, y)

            # Check for a pinch gesture: if the thumb (landmark 4) and index finger tip (landmark 8) are close
            thumb_tip = hand_landmarks.landmark[4]
            if np.abs(index_finger_tip.x - thumb_tip.x) < 0.05 and np.abs(index_finger_tip.y - thumb_tip.y) < 0.05:
                pyautogui.click()

            # Check for a fist gesture (right click)
            if is_fist(hand_landmarks):
                pyautogui.rightClick()  # Perform right-click

            # Check for open hand gesture (scrolling)
            if is_open_hand(hand_landmarks):
                # Here, you can check if the hand is moving up or down to scroll
                # Move the mouse scroll based on y-axis movement
                pyautogui.scroll(5)  # Scroll up or use negative for scrolling down

            # Double click gesture
            is_double_click(hand_landmarks)

            # Dragging gesture
            is_dragging_gesture(hand_landmarks)

            # Volume control based on hand movement
            control_volume(hand_landmarks)

    # Display the frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
