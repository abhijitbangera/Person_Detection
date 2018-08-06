import cv2
import os
import numpy as np
from tkinter import *
import tkinter,os
import cv2
import shutil
from tkinter import filedialog
from PIL import Image, ImageTk


subjects = ["", "Abhijit", "Nikhita","Dev"]

class person_detect():

	def __init__(self):
		self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	def detect_face(self,img):
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
	    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	    if (len(faces) == 0):
	        # cv2.imshow("NO FACE", img)
	        # cv2.waitKey(5000)
	        return None, None
	    (x, y, w, h) = faces[0]
	    return gray[y:y+w, x:x+h], faces[0]

	def prepare_training_data(self,data_folder_path):
	    dirs = os.listdir(data_folder_path)
	    faces = []
	    labels = []
	    for dir_name in dirs:
	        if not dir_name.startswith("person"):
	            continue;
	        label = int(dir_name.replace("person", ""))
	        subject_dir_path = data_folder_path + "/" + dir_name
	        subject_images_names = os.listdir(subject_dir_path)
	        for image_name in subject_images_names:
	            if image_name.startswith("."):
	                continue;
	            image_path = subject_dir_path + "/" + image_name
	            image = cv2.imread(image_path)
	            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
	            cv2.waitKey(100)
	            
	            face, rect = self.detect_face(image)
	            if face is not None:
	                faces.append(face)
	                labels.append(label)
	            
	    cv2.destroyAllWindows()
	    cv2.waitKey(1)
	    cv2.destroyAllWindows()
	    
	    return faces, labels

	def draw_rectangle(self,img, rect):
	    (x, y, w, h) = rect
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	def draw_text(self,img, text, x, y):
	    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	def predict(self,test_img):
		if test_img is None:
			print ("it is none")
			return
		img = test_img.copy()
		face, rect = self.detect_face(img)
		print (face)
		label, confidence = self.face_recognizer.predict(face)
		self.label_text = subjects[label]

		self.draw_rectangle(img, rect)
		self.draw_text(img, self.label_text, rect[0], rect[1]-5)

		return img, self.label_text

	def main(self,img1,img2):
		print (img1)
		path1=os.getcwd() + os.sep + "test-data"+ os.sep+"test1.jpg"
		path2=os.getcwd() + os.sep + "test-data"+ os.sep+"test2.jpg"

		shutil.copy2(img1, path1)
		shutil.copy2(img2, path2) 

		print("Preparing data...")
		faces, labels = self.prepare_training_data("training-data")
		print("Data prepared")
		
		self.face_recognizer.train(faces, np.array(labels))
		print("Predicting images...")

		test_img1 = cv2.imread("test-data/test1.jpg")
		test_img2 = cv2.imread("test-data/test2.jpg")

		predicted_img1, self.predicted_img1_text = self.predict(test_img1)
		predicted_img2, self.predicted_img2_text = self.predict(test_img2)
		print("Prediction complete")
		print (predicted_img1)
		print (predicted_img2)

		if predicted_img1 is None:
			print ("img1 is none")
		if predicted_img2 is None:
			print ("img2 is none")
		out1=subjects[1], cv2.resize(predicted_img1, (400, 500))
		out2= subjects[2], cv2.resize(predicted_img2, (400, 500))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.waitKey(1)
		cv2.destroyAllWindows()
		self.display_images(out1,out2)

	def display_images(self, out1, out2):
		# Load an color image
		img = out1[1]
		#Rearrang the color channel
		b,g,r = cv2.split(img)
		img = cv2.merge((r,g,b))
		# Convert the Image object into a TkPhoto object
		im = Image.fromarray(img)
		image = im.resize((400, 400), Image.ANTIALIAS)
		imgtk = ImageTk.PhotoImage(image=image) 
		# Put it in the display window
		panel = Label(root, image=imgtk)
		panel.pack() 
		panel.configure(image=imgtk)
		panel.image = imgtk
		label1 = tkinter.Label(root, text ="AI predicts the image to be of: "+self.predicted_img1_text)
		label1.pack(side=LEFT)
		label1.place(x=1050,y=200)

		img2 = out2[1]
		#Rearrang the color channel
		b,g,r = cv2.split(img2)
		img2 = cv2.merge((r,g,b))
		# Convert the Image object into a TkPhoto object
		im2 = Image.fromarray(img2)
		image2 = im2.resize((400, 400), Image.ANTIALIAS)
		imgtk1 = ImageTk.PhotoImage(image=image2) 
		# Put it in the display window
		panel2 = Label(root, image=imgtk1)
		panel2.pack(side="right", fill="both") 
		panel2.configure(image=imgtk1)
		panel2.image = imgtk1	
		label1 = tkinter.Label(root, text ="AI predicts the image to be of: "+self.predicted_img2_text)
		label1.pack(side=LEFT)
		label1.place(x=930,y=600)

def take_photos():
	cam = cv2.VideoCapture(0)
	cv2.namedWindow("test")
	img_counter = 0
	
	while True:
	    ret, frame = cam.read()
	    cv2.imshow("test", frame)
	    if not ret:
	        break
	    k = cv2.waitKey(1)

	    if k%256 == 27:
	        # ESC pressed
	        print("Escape hit, closing...")
	        break
	    elif k%256 == 32:
	        # SPACE pressed
	        img_name = "opencv_frame_{}.png".format(img_counter)
	        cv2.imwrite(img_name, frame)
	        print("{} written!".format(img_name))
	        img_counter += 1
	cam.release()
	cv2.destroyAllWindows()


root = tkinter.Tk()
root.title("Person Detection")
#Start - Code to make full screen
root.state('zoomed')
#End
button_dashboard = tkinter.Button(root, text ="Take photos",command= take_photos)
button_dashboard.pack(side=LEFT)
button_dashboard.place(x=30,y=30,relheight=0.10,relwidth=0.10, bordermode="outside")

label1 = tkinter.Label(root, text ="Press space bar to take photos, press ESC to exit from taking photos")
label1.pack(side=LEFT)
label1.place(x=230,y=50)
img1 = None
img2 = None

def browsefunc():
    filename = filedialog.askopenfilename()
    pathlabel.config(text=filename)
    global img1
    img1= filename

def browsefunc2():
    filename = filedialog.askopenfilename()
    pathlabel2.config(text=filename)
    global img2
    img2= filename

browsebutton = tkinter.Button(root, text="Upload Image 1", command=browsefunc)
browsebutton.pack(side=LEFT)
browsebutton.place(x=30,y=120,relheight=0.10,relwidth=0.10, bordermode="outside")

pathlabel = Label(root)
pathlabel.pack()
pathlabel.place(x=230,y=150)


browsebutton2 = tkinter.Button(root, text="Upload Image 2", command=browsefunc2)
browsebutton2.pack(side=LEFT)
browsebutton2.place(x=30,y=220,relheight=0.10,relwidth=0.10, bordermode="outside")

pathlabel2 = Label(root)
pathlabel2.pack()
pathlabel2.place(x=230,y=220)

browsebutton3 = tkinter.Button(root, text="Predict Names", command=lambda:person_detect().main(img1,img2),bg="green")
browsebutton3.pack(side=LEFT)
browsebutton3.place(x=30,y=320,relheight=0.25,relwidth=0.25, bordermode="outside")

root.mainloop()
