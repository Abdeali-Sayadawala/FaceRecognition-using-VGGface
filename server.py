"""
Note:
    This is file is for the host machine which gets the images from the client using imagezmq server
    This file doesn't send images but recieves them from the client and recognize faces according to the dataset provided
"""

"""importing all the important files"""
import cv2
"""Imagezmq is a zmq library to send and recieve images over network"""
from Assets import imagezmq
import numpy as np 
import socket
"""vgg_model is out moduel to detect faces and recognize them"""
import vgg_model
#import pymongo

"""initializing the server using imagezmq"""
server_init = imagezmq.ImageHub(open_port='tcp://*:8008')
hostname = socket.gethostname()
ipaddress = socket.gethostbyname(hostname)
#client = pymongo.MongoClient("mongodb://localhost:27017")
#db = client["Ajna"]
#coll = db["Person"]
"""Below code will show the ipaddress of the server it is using
Use this ip address in the client code to connect to this server"""
print("Ip address of server: "+str(ipaddress))

blank = vgg_model.VggFaceNet()
detect = vgg_model.Detection()

"""infinite loop to recieve image from client and show it on screen
    This loop can be broken by pressing the 'q' button"""
while True:
    """We recieve image and message from the client connected"""
    (msg, frame) = server_init.recv_image()    
    
    faces = detect.detectFace(frame)
    
    for (x, y, w, h) in faces:
        faceimg = frame[y:y+h, x:x+w]
        scores = blank.recognize_from_encodings(faceimg)
        print(scores)
        name = max(scores, key = scores.get)
        print(name)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 1)
    
    """When we stop here we send a reply stop to the client to stop it."""
    if cv2.waitKey(1) & 0xFF == ord("q"):
        server_init.send_reply(b'stop')
        break
    """Send reply to server to keep sending."""
    server_init.send_reply(b'OK')
    cv2.imshow(msg, frame)
    
cv2.destroyAllWindows()        
    
"""This is test/debugging code """
#import vgg_model
#m = vgg_model.VggFaceNet()
#m.create_encodings()
#import pickle
#file = open("encodings.pk", "rb")
#encodings = pickle.load(file)
#for e in encodings:
#    print(e)
#    print(encodings[e])
#img = cv2.imread("1.jpg")
#img = cv2.resize(img, (500,600))
#faces = detect.detectFace(img)
#for (x,y,w,h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
#    faceimg = img[y:y+h, x:x+w]
#    scores = blank.recognize_from_encodings(faceimg)
#    print(scores)
#cv2.imshow("frame", img)    
#vc = cv2.VideoCapture(0)
#while True:
#    ret, frame = vc.read()
#    faces = detect.detectFace(frame)
#    for (x,y,w,h) in faces:
#        faceimg = frame[y:y+h, x:x+w]
#        scores = blank.recognize_from_encodings(faceimg)
##        print(scores)
#        name = max(scores, key = scores.get)
#        print(name)
#        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 1)
#    
#    cv2.imshow("Frame", frame)
#    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#vc.release()
#cv2.destroyAllWindows()
    
    
