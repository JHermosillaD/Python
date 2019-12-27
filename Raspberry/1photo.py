import picamera
import datetime
import time 
date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
with picamera.PiCamera() as camera:
	camera.rotation = 180
	camera.hflip = True
	camera.resolution = (1280,720)
	camera.framerate = 30
	camera.start_preview()
	time.sleep(1.5)
	camera.capture('/home/pi/Documents/Piphotos/'+ date + 'photo.jpg')
	camera.stop_preview()
