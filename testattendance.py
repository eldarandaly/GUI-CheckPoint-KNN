from itertools import count
import sys
import platform
from tkinter import PhotoImage
from tokenize import Ignore
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient,QImage)
from PySide2.QtWidgets import *
from PyQt5.QtCore import pyqtSignal,QThread,QTimer
import cv2
import math
from pandas import array
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import time
import imutils
from imutils.video import VideoStream
import threading
from ui_untitled import Ui_Form
from ui_notfi import Ui_Dialog
import sqlite3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")


class FreshestFrame(threading.Thread):
	def __init__(self, capture, name='FreshestFrame'):
		self.capture = capture
		assert self.capture.isOpened()

		# this lets the read() method block until there's a new frame
		self.cond = threading.Condition()

		# this allows us to stop the thread gracefully
		self.running = False

		# keeping the newest frame around
		self.frame = None

		# passing a sequence number allows read() to NOT block
		# if the currently available one is exactly the one you ask for
		self.latestnum = 0

		# this is just for demo purposes		
		self.callback = None
		
		super().__init__(name=name)
		self.start()

	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=None):
		self.running = False
		self.join(timeout=timeout)
		self.capture.release()

	def run(self):
		counter = 0
		while self.running:
			# block for fresh frame
			(rv, img) = self.capture.read()
			assert rv
			counter += 1

			# publish the frame
			with self.cond: # lock the condition for this operation
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		# with no arguments (wait=True), it always blocks for a fresh frame
		# with wait=False it returns the current frame immediately (polling)
		# with a seqnumber, it blocks until that frame is available (or no wait at all)
		# with timeout argument, may return an earlier frame;
		#   may even be (0,None) if nothing received yet

		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum+1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)

			return (self.latestnum, self.frame)

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown.NoID", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        ID=name.split('.')[1]
        name=name.split('.')[0]
        #print(name,ID)
        # Draw a box around the face using the Pillow module
        if name!='unknown':
           draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        else:
            draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        draw.text((left + 6, bottom - text_height + 5), "ID:"+ID, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui=Ui_Form()
        
        self.ui.setupUi(self)
    

        self.Worker1 = Worker1()
        self.Worker1.start()
      
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        #self.Worker1.signal1.connect(self.display)
        self.Worker1.SendInfoToMainScreen.connect(self.getRecognationInfo)
        self.Worker1.SendInfoToPopUpScreen.connect(self.openscreen)
        #self.Worker1.SendInfoToPopUpScreen.disconnect(self.openscreen)
        
    #def display(self,name,id,img):
       # self.ui.EmpName.setText(name)
        #self.ui.EmpID.setText(id)
        #empphoto=QPixmap(img)
       # self.ui.EmpPhoto.setPixmap(empphoto)
        #self.ui.EmpPhoto.setScaledContents(True)

    def ImageUpdateSlot(self, Image):
        self.ui.feed.setPixmap(QPixmap.fromImage(Image))
        self.ui.feed.setScaledContents(True)

    def CancelFeed(self):
        self.Worker1.stop()   
    
    def getRecognationInfo(self,predictions,counter):
        name=predictions.split('.')[0]
        Id=predictions.split('.')[1]
        if counter==0:
            #--------1
            self.ui.EmpName.setText("Name:"+name)
            self.ui.EmpID.setText("ID:"+Id)
            empphoto=QPixmap(self.ph)
            self.ui.EmpPhoto.setPixmap(empphoto)
            self.ui.EmpPhoto.setScaledContents(True)
        #---------2
        elif counter == 1 :
            self.ui.EmpName_2.setText("Name:"+name)
            self.ui.EmpID_2.setText("ID:"+Id)
            empphoto_2=QPixmap(self.ph)
            self.ui.EmpPhoto_2.setPixmap(empphoto_2)
            self.ui.EmpPhoto_2.setScaledContents(True)

        elif counter == 2 :
            self.ui.EmpName_3.setText("Name:"+name)
            self.ui.EmpID_3.setText("ID:"+Id)
            empphoto_3=QPixmap(self.ph)
            self.ui.EmpPhoto_3.setPixmap(empphoto_3)
            self.ui.EmpPhoto_3.setScaledContents(True)

        elif counter == 3 :
            self.ui.EmpName_4.setText("Name:"+name)
            self.ui.EmpID_4.setText("ID:"+Id)
            empphoto_4=QPixmap(self.ph)
            self.ui.EmpPhoto_4.setPixmap(empphoto_4)
            self.ui.EmpPhoto_4.setScaledContents(True)
        elif counter == 4 :
            self.ui.EmpName_5.setText("Name:"+name)
            self.ui.EmpID_5.setText("ID:"+Id)
            empphoto_5=QPixmap(self.ph)
            self.ui.EmpPhoto_5.setPixmap(empphoto_5)
            self.ui.EmpPhoto_5.setScaledContents(True)
            
        elif counter == 5 :
            self.ui.EmpName_6.setText("Name:"+name)
            self.ui.EmpID_6.setText("ID:"+Id)
            empphoto_6=QPixmap(self.ph)  
            self.ui.EmpPhoto_6.setPixmap(empphoto_6)
            self.ui.EmpPhoto_6.setScaledContents(True)
            
        
    def openscreen(self,predictions):
        self.win=popupScreen()

        conn = sqlite3.connect("./DataBaseTabletest.db") 
        conn.text_factory=str
        cursor = conn.cursor()
        #for Id in predictions:

        name=predictions.split('.')[0]
        Id=predictions.split('.')[1]
        
        cursor.execute("select Emp_Photo from Employees where Emp_ID = (?);", (Id,))
        result = cursor.fetchone()
        self.ph=result[0]

        self.win.popup(self.ph,"Dep Is Here","Job Title Is Here",predictions)
        QTimer.singleShot(2000,self.win.close)
        #self.Worker1.SendInfoToPopUpScreen.disconnect()     



        
        

class popupScreen(QDialog):
    
    def __init__(self):
        super(popupScreen, self).__init__()
        self.ui=Ui_Dialog()
        self.ui.setupUi(self)

    def popup(self,img,dep,title,predictions):
        empphoto=QPixmap(img)
        self.ui.EmpPhoto.setPixmap(empphoto)
        self.ui.EmpPhoto.setScaledContents(True)
        #
        name=predictions.split('.')[0]
        empID=predictions.split('.')[1]
        self.ui.EmpName.setText(str(name))
        self.ui.EmpID.setText(str(empID))
        self.ui.EmpDep.setText(str(dep))
        self.ui.EmpJobTitle.setText(str(title))
        self.show()
        
     
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    #signal1= pyqtSignal(str,str,str)
    SendInfoToMainScreen=pyqtSignal(str,int)
    SendInfoToPopUpScreen=pyqtSignal(str)
    
    def run(self):
        self.ThreadActive = True
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        #video_capture = cv2.VideoCapture("rtsp://admin:TZZUNI@192.168.1.58:554/H.264", cv2.CAP_FFMPEG)
        #video_capture = cv2.VideoCapture("http://192.168.1.54:8080/video")
        video_capture=cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FPS, 30) 
        fresh = FreshestFrame(video_capture) 
        process_this_frame=39   
        predictionsList=[]
        counter=0
        while True:
            try:
                ret, Image = video_capture.read()
                
                timer =time.time()
                #Image = cv2.resize(Image, (0, 0), fx=1, fy=1)  
                Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(Image,cv2.COLOR_RGB2GRAY)
                #ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                #self.ImageUpdate.emit(ConvertToQtFormat)
                #
                
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                process_this_frame = process_this_frame + 1
                for (x,y,w,h)in faces:
                    face=Image[y-5:y+h+5,x-5:x+w+5]
                    resized_face=cv2.resize(face,(160,160))
                    resized_face = resized_face.astype("float") / 255.0
                    resized_face = np.expand_dims(resized_face, axis=0)
                    preds = model.predict(resized_face)[0]
                    if process_this_frame % 40 == 0 :
                        predictions = predict(Image, model_path="trained_knn_modelOneShot1.clf")      
                        if predictions:
                            self.SendInfoToPopUpScreen.emit(predictions[0][0])
                            self.SendInfoToMainScreen.emit(predictions[0][0],counter) 
                            counter+=1
                            if counter==6:
                                counter=0
                            #Image= show_prediction_labels_on_image(Image, predictions)
                    
                    cv2.putText(Image, predictions[0][0], (x,w),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(Image, (x, y), (x+w,y+h),(0, 255, 0), 2)
                    if preds>0.8:
                        label = 'spoof'
                        FakeFlage=True
                        RealFlage=False
                        cv2.putText(Image, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        #cv2.rectangle(Image, (x, y), (x+w,y+h),(0, 0, 255), 2)
                    else:
                        label = 'real'
                        FakeFlage=False
                        RealFlage=True
                        cv2.putText(Image, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        #cv2.rectangle(Image, (x, y), (x+w,y+h),(0, 255, 0), 2)
                
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                
                self.ImageUpdate.emit(ConvertToQtFormat)
                path='./images/train/ahmed.3/img (1).jpeg'
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.putText(Image,f'FPS:{int(fps)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)

                #self.SendInfoToMainScreen.emit(predictions[0][0],counter) 
                
                #print('here')
                #self.ImageUpdate.emit(ConvertToQtFormat)
            except Exception as e:
                pass        

       


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())