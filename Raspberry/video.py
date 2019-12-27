import datetime
import time
date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
import picamera
with picamera.PiCamera() as camera:
    camera.rotation = 180
    camera.start_preview()
    camera.start_recording("/home/pi/Desktop/video/"+ date + "video.h264")
    camera.wait_recording(15)
    camera.stop_recording()
    camera.stop_preview()
