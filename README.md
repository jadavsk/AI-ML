# AI-ML
Learn AI and ML 




Project Goal:
To Ensure Safe Office Environment, employees should cover their face with Mask and maintain Social distance to limit the spread of the Corona virus. This is essential to implement post-Covid normal working in office. Identify employees who are not wearing face Mask and following social distancing while entering office or ODC floor.

Solution: 
Research estimates that wearing a face mask may reduce the transmission of COVID-19 by up to 80%. Wearing a cloth face covering and maintain Social distance of 6-ft will help protect people around, including those at higher risk of severe illness from COVID-19 and colleague who frequently come into close contact with others in office environment. Implement video analytics solution, leveraging latest AI/ML technology having below features:
Identify employees who are not wearing face Mask using ML-Face Mask detection and recognition while entering office building or ODC floor.
Notification Email to Employee and HR/Admin if employee is recognized without Mask (Warning/Alert).
Integrate Face Masks detection system with door swipe badging (Card Access Systems) at the entrance in order to automatically block employees without mask. 
Integrate automatic Voice/Speaker announcement in public Office areas and cafeteria if Social Distancing is not maintained.
Monitor social distancing in specific office areas (like lobbies, cafeteria or ODC floor)

Technical Solution: 
Open Source Machine Learning Components - Convolutional Neural Networks (CNN), YOLO, Python, Keras, Tensorflow , OpenCV and pywin32.

**Face Mask Detection 
Program will detect if Employee is wearing Mask or not.
Program can be connected to any existing or new IP or CCTV cameras to detect employees without a mask while entering office gate or ODC door.
Python program will be trained to detect COVID-19 face mask using OpenCV, Keras/TensorFlow, and Deep Learning.
1000+ Images are used to train the model so that it can accurately identify face mask.
Deploy the trained face mask detector and classify each face as with_mask or without_mask

**Face Recognition
If any Employee is not wearing mask, Face Recognition System will identify the Employee name and log violation date in the system.
Using existing Employee HR database we can create Image Dataset for all employees.
Using the dlib facial recognition network, the output feature vector is 128-d will be created to quantify the face for each Employees
