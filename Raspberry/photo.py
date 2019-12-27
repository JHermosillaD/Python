from picamera import PiCamera
import time
camera = PiCamera()  
camera.rotation = 180
camera.start_preview()
time.sleep(3)
camera.capture('/home/pi/image.jpg')
camera.stop_preview()
