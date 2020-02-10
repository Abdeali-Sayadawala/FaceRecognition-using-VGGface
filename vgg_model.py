"""
This is the library for face recognition using VGGFace network. Here we have functions:
    1.) During the initialization of the library, that is in the __init()__ funtion we create the Convolutional 
                Neural Network of VGGFace and load the weights in the network. Then, if the input for load_encodings
                is 1 then we pre-load the encodings that can be used later for comparing and calculating difference.
                
    2.) Create Encodings: we provide the path of dataset containing images of every person in a folder named 
                according to the name of the person. In this part we pass all the images in our dataset throught the 
                network and save their output in a dictionary with key as theri name. we create encodings so that
                we dont have to pass the images from our dataset everytime while recognizing a face. we just have 
                to pass the face from the network once and compare it with the encodings to get the persons name.
                
    3.) Get Sim: This function is the get similarity function that returns the similarity of both the images passed
                images. It has two methods for calculating similarity (i) Euclidean Distance and (ii) Cosine Similarity
                the user passes the name of the method that should be used as the last parameter. By default it is 
                Euclidiean
    4.) Recognize from encodings: The recognize from encodings function compares the passed image from all the 
                encodings made from the images of our dataset.
"""

""" We import all the necessary files for face recognition. """
import os
import cv2
import pickle
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import image as kimg
from keras.applications import imagenet_utils
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Input, Activation
from keras.models import Model, Sequential

"""
Below is two functions of the two methods that is euclidean distance and cosine similarity. we pass two 
representations in the function and get the distance.
Representations are the encodings of the image which we obtain when we pass the image from the network.
"""
def get_cosine_sim(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, test))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def get_euclidean_dist(source, test):
    dist = source - test
    dist = np.sum(np.multiply(dist, dist))
    dist = np.sqrt(dist)
    return dist

class Detection():
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier("Assets/haarcascade_frontalface_default.xml")
    
    def detectFace(self, image):        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,        
                minSize=(30,30)
                )
        return faces
    
    
class VggFaceNet(object):
    epsilon_cosine = 0.40
    epsilon_euclid = 80
    
    """ We make the network in the init function and due to this the network gets preloaded when the object is created"""
    def __init__(self, load_encodings=1):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
            
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(Conv2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        
        model.load_weights("vgg_face_weights.h5")
        
        """Here we load all the encodings from the file where the created encodings are kept.
        But if the load_encodings parameter is manually kept 0 the program doesn't load the encodings.
        This can be done if you simply want to compare two images using the get_values and get_sim functions.
        We load this now on the creation of object so less time is consumed when recognizing"""
        self.__vggNet = Model(inputs = model.layers[0].input, outputs = model.layers[-2].output)
        self.load_encodings = load_encodings
        if load_encodings == 1:
            if os.path.exists("encodings.pk"):
                file = open("encodings.pk", "rb")
                self.encodings_from_file = pickle.load(file)
                file.close
                self.scores = {}
                for x in self.encodings_from_file:
                    self.scores[x] = 0
            else:
                print("Couldn't find encodings please create using the create_encodings function.")
        
    def process_img(self, img):
        """
        img: We pass the image we read using opencv
        This function processes the image according to the need of VGGFace network.
        """
        img = cv2.resize(img, (224, 224))
        img = kimg.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = imagenet_utils.preprocess_input(img)
        return img
    
    def get_values(self, image):
        """
        image: we passed the processes image using the function process_img
        This function predicts the features/encodings of the image passed
        """
        return self.__vggNet.predict(image)
    
    def get_sim(self, representation1, representation2, diff_type="euclidean"):
        """
        representation1: We pass the first representation we got from passing 1st image through the network
        representation2: We pass the second representation we got from passing 2nd image through the network
        diff_type: we pass he name of the method we want to use.
        Here we calculate the similarity of the two representations by using either euclidean or cosine methods.
        ***The cosine method is not working for a reason and in future updates as soon as it starts working this
        starred comment will be removed.***
        """
        
        if diff_type == "euclidean":
            distance = get_euclidean_dist(representation1, representation2)
        else:
            distance = get_cosine_sim(representation1, representation2)
            
        return distance
    
    def create_encodings(self, data = "Dataset"):
        """
        data: We pass the file path of the dataset of the images
        in this function we create the encodings of the images present in our dataset. 
        If the encodings are created we overwrite them.
        Don't use the same images more than once for the same person.
        """
        if os.path.exists("encodings.pk"):
            file = open("encodings.pk", "rb")
            encodings_from_files = pickle.load(file)
            if len(os.listdir(data)) != 0:
                for person in os.listdir(data):
                    count = 1
                    if len(os.listdir(os.path.join(data, person))) != 0:
                        entry = []
                        for img in os.listdir(os.path.join(data, person)):
                            print("Image "+str(count)+"/"+str(len(os.listdir(os.path.join(data, person))))+" -- "+str(person))
                            image_to_encode = cv2.imread(str(data)+"/"+str(person)+"/"+str(img))
                            entry.append(self.get_values(self.process_img(image_to_encode)))
                        encodings_from_files[person] = entry
                    else:
                        print("no image found for "+str(person))
            else:
                print("No data found in folder..")
            file.close()
        else:
            encodings = {}
            if len(os.listdir(data)) != 0:
                for person in os.listdir(data):
                    count = 1
                    if len(os.listdir(os.path.join(data, person))) != 0:
                        entry = []
                        for img in os.listdir(os.path.join(data, person)):
                            print("Image "+str(count)+"/"+str(len(os.listdir(os.path.join(data, person))))+" -- "+str(person))
                            image_to_encode = cv2.imread(str(data)+"/"+str(person)+"/"+str(img))
                            entry.append(self.get_values(self.process_img(image_to_encode)))
                        encodings[person] = entry
                    else:
                        print("no image found for "+str(person))
            else:
                print("No data found in folder..")
            file = open("encodings.pk", "wb")
            pickle.dump(encodings, file)
            file.close()
        file = open("encodings.pk", "rb")
        self.encodings_from_file = pickle.load(file)
        file.close()
        self.scores = {}
        for x in self.encodings_from_file:
            self.scores[x] = 0
            
    def recognize_from_encodings(self, cam_image):
        """
        cam_image: We pass the processed image in this parameter
        This function compares the representations of image passed with the encodings we created.
        """
        if self.load_encodings == 0:
            print("Encodings are not preloaded")
            ans = int(input("Do you wish to load encodings (1/0): "))
            if ans == 1:
                file = open("encodings.pk", "rb")
                self.encodings_from_file = pickle.load(file)
                file.close
                self.scores = {}
                for x in self.encodings_from_file:
                    self.scores[x] = 0
            else:
                print("Encodings not loaded")
                return 0
        encodings_image = self.get_values(self.process_img(cam_image))
        positive_count = 0
        for person in self.encodings_from_file:
            for enc in self.encodings_from_file[person]:
                dist = self.get_sim(enc, encodings_image)
                if dist < self.epsilon_euclid:
                    positive_count = positive_count + 1
                self.scores[person] = positive_count
            positive_count = 0
        return self.scores