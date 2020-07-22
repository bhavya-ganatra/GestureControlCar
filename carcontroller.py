import numpy as np
import cv2
import pyautogui
#from directkeys import PressKey, ReleaseKey, W, A, S, D
import time
from datetime import datetime
from PIL import ImageGrab
import os

filename = 'video.avi'
filename2 = 'cap.avi'
res = '480p'
res2= '1080p'

def change_res(cap, width, height):
    print(width,height,"------------")
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device to the resulting resolution
    change_res(cap, width, height)
    return width, height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


time.sleep(4)
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))
#out2 = cv2.VideoWriter(filename2, get_video_type(filename2), 25, (1000,575)) #set dimension acording to bbox=(300,115,1300,690)
out2 = cv2.VideoWriter(filename2, get_video_type(filename2), 25,STD_DIMENSIONS[res2])

count=0
while(True):
        _, frame = cap.read()
        #img_rec = np.array(ImageGrab.grab(bbox=(300,115,1300,690)))
        img_rec = np.array(ImageGrab.grab(bbox=None))
        img_rec = cv2.cvtColor(img_rec,cv2.COLOR_RGB2BGR)
        print(img_rec.shape)
        count+=1
        #if(count%3==0):
        #print(count)
        print(datetime.now(), count)
        frame = cv2.resize(frame,STD_DIMENSIONS[res])#,fx=2,fy=2,interpolation=cv2.INTER_AREA)
        #print(frame.shape)
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.rectangle(frame,(int(w/1.6), int(h/3.75)),(w, int(h/1.2)),(255,0,0),3) #RIGHT
        cv2.putText(frame,'RIGHT',(int(w/1.6)+5, int(h/3.75)+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2)

        cv2.rectangle(frame,(0,int(h/3.75)),(int(w/2.5),int(h/1.2)),(255,0,0),3)     #LEFT
        cv2.putText(frame,'LEFT',(0+5,int(h/3.75)+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2)

        cv2.rectangle(frame,(int(w/4), int(h/1.7)),(int(w/1.3), int(h)),(255,0,0),3) #BOTTOM
        cv2.putText(frame,'DOWN',(int(w/4)+5, int(h/1.5)+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2)

        cv2.rectangle(frame,(int(w/4), 0),(int(w/1.3), int(h/2.5)),(255,0,0),3)   #TOP
        cv2.putText(frame,'UP',(int(w/4)+5, 0+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2)


        final_lb = np.array([0,98,86])
        final_ub = np.array([7,237,192])
        mask = cv2.inRange(hsv, final_lb, final_ub)    
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cnts,hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        try:
            cnt = max(cnts, key = cv2.contourArea)
            #print(cv2.contourArea(cnt))
        except:
            pass
        if(len(cnts)!=0 and cv2.contourArea(cnt)>500):

            x,y,w_c,h_c = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w_c,y+h_c),(0,255,0),2)

            right = [[int(w/1.6), int(h/3.75)],[w, int(h/1.2)]] #RIGHT
            left = [[0,int(h/3.75)],[int(w/2.5),int(h/1.2)]]     #LEFT
            bottom = [[int(w/4), int(h/1.7)],[int(w/1.3), int(h)]] #BOTTOM
            top = [[int(w/4), 0],[int(w/1.3), int(h/2.5)]]   #TOP

            if((x,y) < (left[1][0],left[0][1])):      #in left region
                pyautogui.keyDown('a')
                pyautogui.keyUp('a')
                #PressKey(A)
                #ReleaseKey(A)
                #print("Left")

            elif((x+w_c,y) > (right[0][0],right[0][1])):     #in Right region
                pyautogui.keyDown('d')
                pyautogui.keyUp('d')
                #PressKey(D)
                #ReleaseKey(D)
                #print("Right")

            if(y < top[1][1]):       
                pyautogui.keyDown('w')
                pyautogui.keyUp('w')
                #PressKey(W)
                #ReleaseKey
                #print("Up")

            elif(y+h_c > bottom[0][1]):
                pyautogui.keyDown('s')
                pyautogui.keyUp('s')
                #PressKey(S)
                #ReleaseKey(S)
                #print("Down")

            cv2.imshow("mask", mask)
            cv2.imshow("Result", result)
        out.write(frame) 
        out2.write(img_rec)
        cv2.imshow("Original Frame",frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
cap.release()
out.release()
out2.release()
cv2.destroyAllWindows()
