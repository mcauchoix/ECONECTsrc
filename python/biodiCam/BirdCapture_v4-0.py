# Functional script that scans low resolution images
# and records a burst of images when the difference between
# two consecutive images exceeds a certain threshold.

import time
import picamera
import numpy as np
import cv2
from threading import Thread

BURST = 8        # Number of pics per burst
LATENCY = 4.0       # Minimal time (in sec.) between two bursts
WIDTH = 1920      # Horizontal resolution of recorded pics
HEIGHT =1088     # Vertical resolution of recorded pics
FPS = 12          # Camera frame rate
THRESHOLD = 5.0  # % change in pixels to detect motion
CHECK_INTERVAL = 1 # Interval of time (in sec.) between cam status checking

counter = 0
percent_diff = 0.0
filenames_list = []
cam_state = "--"
info_stop_toDisp = True

camera = picamera.PiCamera()
camera.resolution = (WIDTH, HEIGHT)
camera.framerate = FPS

class check_status(Thread):
    def __init__(self,time_interval):
        Thread.__init__(self)
        self.time_interval =  time_interval
        self.last_check = time.time()    
    def run(self):
        global cam_state
        cam_state = "running"
        
        print("Thread running...")
        
        fichier = open("/var/www/cgi-bin/cam_state.txt", "r")
        cam_state = fichier.read()
        fichier.close()
        
        while True:
            now = time.time()
            if now - self.last_check > self.time_interval:
                self.last_check = now
                fichier = open("/var/www/cgi-bin/cam_state.txt", "r")
                cam_state = fichier.read()
                fichier.close()

def displayInfo():
    global lastDisp
    if now-lastPic_time > LATENCY:
        print("Diff. between pics: " + str(round(percent_diff,1)) + "%")
    else:
        timeBeforeResume = round(LATENCY - (now-lastPic_time))
        if now - lastDisp >= 1:
            print("Pause between bursts (" + str(timeBeforeResume) + " sec. before resume)")
            lastDisp = now

def recordBurst():
    global filenames_list, lastPic_time, percent_diff
    time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    print("Pic recorded (" + time_stamp + ")!")
    filename = "/var/www/html/img/biodicam/" + time_stamp + "_(" + str(round(percent_diff,0)) + ").jpg"
    cv2.imwrite(filename, frame)
    print("Camera status: " + cam_state)
    #filenames_list = []
    lastPic_time = now
    percent_diff = 0.0

time.sleep(2)
now = time.time()
lastPic_time = now
lastDisp = now

frame0 = np.empty((1088, 1920, 3), dtype=np.uint8)
frame = np.empty((1088, 1920, 3), dtype=np.uint8)
camera.capture(frame0, 'bgr')

frame0_light = cv2.resize(frame0,(192,108),interpolation = cv2.INTER_AREA)

thread_statusCheck = check_status(CHECK_INTERVAL)
thread_statusCheck.start()

while True:
    if cam_state == "record":
        info_stop_toDisp = True
        now = time.time()
        if percent_diff > THRESHOLD and now - lastPic_time > LATENCY:
            recordBurst()
            
        else:
            camera.capture(frame, 'bgr')
            frame_light = cv2.resize(frame,(192,108),interpolation = cv2.INTER_AREA)	
            diff = cv2.absdiff(frame0_light,frame_light) 
            img_black = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            percent_black = round(np.sum(img_black < 10)/(192*108),2)
            percent_diff = (1-percent_black)*100

        displayInfo()
        frame0_light = frame_light
        
    else:
        if info_stop_toDisp == True:
            print("Camera stopped")
            info_stop_toDisp = False
