from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2, threading

# Create your views here.

def home(request):
    return render(request, 'webHome.html')

