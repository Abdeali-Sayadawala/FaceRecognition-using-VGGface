"""
Abdeali
02-12-2019, 11:10
"""

"""importing all important files"""
import cv2
import imagezmq
import socket
import time

"""Connecting to the ip address of server
	Change the server IP address every time you turn on the server
	The server file will show the ip address it is using when it starts"""
server_address = imagezmq.ImageSender(connect_to="tcp://192.168.0.6:8008")

"""Camera Setup"""
vc = cv2.VideoCapture(0)
print("Setting up camera", end="")
for i in range(5):
    time.sleep(0.4)
    print(".", end="")

"""Infinite loop to capture and send frames till 'q' presses on keyboard"""
ret, frame = vc.read()
while True:
    """Send image captured from camera to server"""
    msg = server_address.send_image("1", frame)
    print(msg)
        
    """if the server replies to stop the we stop the loop and break out"""
    if msg == b'stop' or msg == 'stop':
        break
    
    """Buffer wait: We wait for 2 second before reading another frame and sending it to the server
    because there is a lag in recieving the video at server end and sending frames continuously
    is a very heavy and slow task and reduces life of device. Also there will be time consuming task
    at server as the server will perform face recognition and then put an entry in the database log
    for the person deteted. Uncomment the code if you want to use Buffer wait."""
    #time.sleep(2)
    ret, frame = vc.read()
    #cv2.imshow("USB cam 1", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
"""Release the camera from process and destroy the windows showing video"""
vc.release()
cv2.destroyAllWindows() 