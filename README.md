# Eigenface-facial-recognition-in-Python-during-a-live-feed

This project is a implementation of the famous Eigenface face recognition algorithm in Python that works in a live-feed via the users webcam. 
During runtime the program will attempt to find and classify a person in the current frame, people in the training set will have their names applied to the frame whilst new 
users will have the string "Unknown face" applied.

The training data can be interchanged with new faces and have more images added as needed, the program will adapt and use the new names as labels.

Since the program isn't computationally demanding it should work in most/all IDEs and on most systems.
