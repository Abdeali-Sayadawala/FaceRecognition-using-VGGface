Link for VGGFace weights : https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing

This repository has a vggmodel.py file that uses Vggface network and its pretrained weights to recognize faces given in the dataset
or provided manually

The server.py file is a server used to get images from the client, detect faces in it and recognize it. It only works on local server
you can modify to code according to your needs if you want to use it on online server 

the client.py file in the Client Code directory gets the frames from the camera and sends it to the server created by server.py file 

Read the instructions given in both the files carefully to know how to use them. 

Both the server.py and the client.py file uses imagezmq library that is created by jeffbass. 

imagezmq: https://github.com/jeffbass/imagezmq

This is the link to the original repository and this is used by me for transferring images from client to server