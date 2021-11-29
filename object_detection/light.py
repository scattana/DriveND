import RPi.GPIO as GPIO
import time
import signal
import sys

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

lightPin = 18
buzzer = 17
button = 4
green = 26
red = 16

GPIO.setup(lightPin, GPIO.OUT)
GPIO.setup(button, GPIO.IN)
GPIO.setup(green, GPIO.OUT)
GPIO.setup(red, GPIO.OUT)
GPIO.setup(buzzer, GPIO.OUT)
pb = GPIO.PWM(buzzer, 500)
pb.start(0)

def buttonExit(pin):
	cleanup()

def interruptExit(key, time):
	cleanup()

def soundOn(pin):
	pin.ChangeDutyCycle(15)

def soundOff(pin):
	pin.ChangeDutyCycle(0)

def lightOn():
	GPIO.output(lightPin, GPIO.HIGH)

def lightOff():
	GPIO.output(lightPin, GPIO.LOW)

def blink(pin):
	pin.start(25)

def redOnStartup():
    GPIO.output(red, GPIO.HIGH)

def redOffStartup():
    GPIO.output(red, GPIO.LOW)

def greenOnStartup():
    GPIO.output(green, GPIO.HIGH)

def greenOffStartup():
    GPIO.output(green, GPIO.LOW)

def cleanup():
	print 'Exiting Program'
	soundOff(pb)
	lightOff()
	redOffStartup()
	greenOffStartup()
	sys.exit()

GPIO.add_event_detect(button, GPIO.RISING, callback=buttonExit)
signal.signal(signal.SIGINT, interruptExit)
