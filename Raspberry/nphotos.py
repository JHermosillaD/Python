import picamera
import time 
with picamera.PiCamera() as camera:
	camera.rotation = 180
	camera.hflip = True
	camera.resolution = (640,480)
	camera.framerate = 30
	camera.start_preview()
	for i in range(30):
		time.sleep(120)
		camera.capture('/home/pi/Documents/Piphotos/Toma1/image%s.jpg' % i)
	camera.stop_preview()
