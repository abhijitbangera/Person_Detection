# Person Identification using Opencv library

Person detection program learns to identify a individual by learning on the training data provided.
It then is able to predict the person correctly when a new image of the person is provided as a test data.

### Primary libraries used:
1. opencv2
2. tkinter
3. PIL

### Folder structure:
There are folders:
1. training-data
2. test-data

Inside training data we need to create folders with prefix "person". Example: "person1" folder can contain my images for it to train. "person2" folder may contain a second person's images and so on.

### How to run
Once all the required libraries are installed on the system, just open the command prompt in the directory and execute the below command and rest is the GUI.
'''
person_detection_opencv.py
'''

Now it will open the GUI. You need to upload the 2 images by selecting upload image 1 and upload image 2 button. Once done, click on predict.
I have added a feature to take photos from your webcam so that it will be easy to create training data. Make sure image format in training-data is jpg format.

### Demo:
![Alt Text](https://github.com/abhijitbangera/Person_Detection/blob/master/DEMO_person_detection_gui.gif)
