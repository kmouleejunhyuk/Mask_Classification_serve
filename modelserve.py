#py for model lightweighting & inferencing
#input: img(facecropped)
#output: label, alert/pass 여부
import io
from typing import Tuple
import torchvision.transforms as transforms
import torch
import os

class modelserve():
    def __init__(self, quantize: bool = True, model_dir: str = r'./model_mnist1.pickle', imagesize: Tuple=(1, 3, 584, 312)):
        self.modeldir = model_dir
        self.quantize = quantize
        self.mean = (0.5601, 0.5241, 0.5014)
        self.std = (0.2331, 0.2430, 0.2456)
        self.imagesize = imagesize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        


    def trans_image(self, image):
        img = self.transform(image).unsqueeze(0)
        return img


    def predict(self, image):
        tensor_image = self.trans_image(image)
        onehot_label = self.model(tensor_image)
        return torch.argmax(onehot_label)


    def load_model(self):
        '''
        if compile = True, complie model pickle and load model 
        else, load compiled model
        '''
        
        if not any([x in self.modeldir for x in ['.pt', '.pickle']]) or not os.path.isfile(self.modeldir):
            raise Exception('Wrong file name. please check')

        else:   #if filedir is valid
            try:
                # cuda 를 사용할 수 없는 device 의 경우 cpu 로 모델을 불러옴
                if self.device == 'cuda':
                    model = self.quantize_model(torch.load(self.modeldir))
                else:
                    model = self.quantize_model(torch.load(self.modeldir, map_location=torch.device(self.device)))
                return model

            except:
                raise Exception('model load failed')


    def quantize_model(self, model):
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        if self.quantize:
            #quantizing model is in beta. may not work properly
            return torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
        else:
            return model

            

#testcode
from PIL import Image
import time

serve = modelserve()
frame = Image.open(f'test.jpeg')
# frame = Image.open(r"P:\Downloads\face.png")
normalmodel = torch.load('model_mnist1.pickle', map_location=torch.device(serve.device))

st = time.time()
p = torch.argmax(normalmodel(transforms.ToTensor()(frame).unsqueeze(0)))
normaltime = time.time() - st

st = time.time()
label = serve.predict(frame)
qtime = time.time()-st

print(p, label, normaltime, qtime)
        