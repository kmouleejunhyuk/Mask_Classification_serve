from django.urls import path
from . import views

app_name = 'webcam_test'

urlpatterns = [
    path('', views.home, name='home'),
    #path('getcam', views.get_cam, name='get_cam'),
]