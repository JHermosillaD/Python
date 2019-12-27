import datetime
import time
date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
import picamera
with picamera.PiCamera() as camera:
    camera.resolution = (1280,720)
    camera.framerate = 30
    camera.rotation = 180
    camera.hflip = True
    camera.start_preview()
    camera.start_recording("/home/pi/Documents/Pivideos/"+ date + "video.h264")
    camera.wait_recording(10)
    camera.stop_recording()
    camera.stop_preview()
