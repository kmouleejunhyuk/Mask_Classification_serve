from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np
import pyshine as ps
from flask_cors import CORS,cross_origin
import imutils
#import dlib
from engineio.payload import Payload


#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*' )



@app.route('/', methods=['POST', 'GET'])

def index():
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):

    emit('response_back', data)  


from modelserve import modelserve
from cam import facecrop
serve = modelserve()
cropper = facecrop()

@socketio.on('image')
def image(data_image):

    #get image from user webcam
    frame = (readb64(data_image))   #userframe
    #crop face from frame
    face = cropper.cropface(frame)
    #label image
    #not implementing label stable
    if isinstance(face, str):
        text = face
    else:
        text  =  'Label: '+str(serve.predict(image = face).item())

    #write image to label
    frame = ps.putBText(frame,text,text_offset_x=20,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back 
    emit('response_back', stringData)



if __name__ == '__main__':
    import flask_cors; flask_cors.CORS(app, resources={r"/*":{"origins":"*"}})
    socketio.run(app, port=8000 ,debug=True)