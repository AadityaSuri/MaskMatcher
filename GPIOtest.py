import RPi.GPIO as GPIO
import time
from gpiozero import Servo

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

REDLED = 22
GREENLED = 27


GPIO.setup(REDLED, GPIO.OUT)
GPIO.setup(GREENLED, GPIO.OUT)

servo = Servo(17)

val = -1


while True:
    servo.value = val
    time.sleep(0.1)
    val = val + 0.1
    if val > 1:
        val = -1
        break;


while True:
    servo.min()
    time.sleep(1)
    servo.mid()
    time.sleep(1)
    servo.max()
    time.sleep(1)
    
    print("LED on")
    GPIO.output(REDLED,GPIO.HIGH)
    GPIO.output(GREENLED,GPIO.LOW)
    time.sleep(1)
    print("LED off")
    GPIO.output(REDLED,GPIO.LOW)
    GPIO.output(GREENLED,GPIO.HIGH)
    time.sleep(1)

