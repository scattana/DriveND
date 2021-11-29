import RPi.GPIO as GPIO
import time
import math

green = 26
red = 16
    
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(green, GPIO.OUT)
GPIO.setup(red, GPIO.OUT)

def redOnStartup():
    GPIO.output(red, GPIO.HIGH)

def redOffStartup():
    GPIO.output(red, GPIO.LOW)

def greenOnStartup():
    GPIO.output(green, GPIO.HIGH)

def greenOffStartup():
    GPIO.output(green, GPIO.LOW)


