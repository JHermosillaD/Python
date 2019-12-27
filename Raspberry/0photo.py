import picamera
import time

with picamera.PiCamera() as camera:
	camera.resolution = (1280,720)
	camera.framerate = 30
	camera.rotation = 180
	camera.hflip = True
	camera.start_preview()
	time.sleep(10)
	camera.stop_preview()
