This project implements a real-time hand gesture recognition system using computer vision techniques. It allows users to interact with computers or applications through hand gestures, eliminating the need for traditional input devices like a mouse or keyboard.

The system captures video input from a webcam, processes hand movements, and maps them to specific actions such as cursor control, clicking, or system commands.

Features--------------------------------->

🎥 Real-time hand tracking using webcam
✋ Gesture recognition (e.g., open palm, fist, finger movement)
🖱️ Mouse control using hand gestures
👆 painting 
⚡ Fast and efficient processing


Technologies Used--------------------->
Python
OpenCV
MediaPipe (for hand tracking)
NumPy
PyAutoGUI (for system control)

System Architecture------------------->

Video Capture
Webcam captures live video frames
Hand Detection
MediaPipe detects hand landmarks
Gesture Recognition
Finger positions are analyzed
Gestures are classified


Usage-------------------------------->
Move your hand in front of the webcam
Use gestures to control the system:
👉 Index finger → Move cursor
✌️ Two fingers → chnage the mode
Make sure lighting conditions are good for better accuracy  


