from django.shortcuts import render
import cv2, threading

# Create your views here.

def main(request):
    return render(request)

class VideoCamera(object):

    def __init__(self):
        self.vedio = cv2.VideoCapture(0)