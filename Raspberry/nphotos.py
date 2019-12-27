from picamera import PiCamera
import time 
camera.start_preview()
camera.rotation = 180
for i in range(5):
    time.sleep(3)
    camera.capture('/home/pi/image%s.jpg' % i)
camera.stop_preview()
