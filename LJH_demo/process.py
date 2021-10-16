import io
import base64,cv2
import numpy as np
import pyshine as ps
from PIL import Image
from cam import facecrop
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from engineio.payload import Payload

Payload.max_decode_packets = 2048
cropper = facecrop()
net=cv2.dnn.readNet("Model.onnx")


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


def normalize(blob, mean = 0.5, std = 0.2):
    blob -= mean * 255
    blob /= std * 255
    return blob


@socketio.on('image')
def image(data_image):
    #not implementing stablizer
    frame = (readb64(data_image))   #get image from user webcam
    # face = cropper.cropface(frame)  #crop face from frame -> face: numpy image(HWC) 0~255

    # if isinstance(face, str):
    #     text = face #if low confidence: NO Face 
    # else:
    #     blob = cv2.dnn.blobFromImage(face, swapRB=False, crop=False, size = (384, 512))
    #     blob = normalize(blob)  #normalize image

    #     net.setInput(blob)
    #     label = np.array(net.forward()) #foward blob
    #     text  =  'Label: '+str(np.argmax(label[0]))
    
    text = 'no DL'
    frame = ps.putBText(frame,text,text_offset_x=20,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0, background_RGB=(10,20,222),text_RGB=(255,255,255))    #write label to image
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY,40])[1]

    stringData = base64.b64encode(imgencode).decode('utf-8')    # base64 encode
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    emit('response_back', stringData)   # emit the frame back 



if __name__ == '__main__':
    import flask_cors; flask_cors.CORS(app, resources={r"/*":{"origins":"*"}})
    socketio.run(app, port=8000 ,debug=True)