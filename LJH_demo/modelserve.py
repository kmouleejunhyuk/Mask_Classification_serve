#py for model lightweighting & inferencing
#input: img(facecropped)
#output: label, alert/pass 여부
import io
from typing import Tuple
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from torchvision.transforms.transforms import CenterCrop, RandomHorizontalFlip

class modelserve():
    def __init__(self, quantize: str = 'bfloat16', model_dir: str = r'./model.pickle', imagesize: Tuple=(1, 3, 584, 312)):
        self.modeldir = model_dir
        self.quantize = quantize
        self.mean = (0.5601, 0.5241, 0.5014)
        self.std = (0.2331, 0.2430, 0.2456)
        self.imagesize = imagesize
        self.device = "cpu"
        self.model = self.load_model()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            #transforms.Resize(100),
                                            #transforms.RandomHorizontalFlip(1)])
                                            transforms.Normalize(self.mean, self.std)])
        


    def trans_image(self, image):
        img = self.transform(image).unsqueeze(0)
        return img


    def predict(self, image):
        with torch.no_grad():
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
                model = torch.load(self.modeldir, map_location=self.device)
                model = self.quantize_model(model)
                return model

            except:
                raise Exception('model load failed')


    def quantize_model(self, model):
        '''
        optional. use to save quantize model
        '''
        for layer in model.parameters():
            layer.requires_grad_(False)
        model.eval()

        if self.quantize == 'qint8':
            #quantizing model is in beta. may not work properly
            return torch.quantization.quantize_dynamic(model, dtype=torch.qint8)

        elif self.quantize == 'bfloat16':
            for layer in model.parameters():
                if not isinstance(layer, torch.nn.BatchNorm2d):
                    layer.bfloat16()
            return model

        else:
            return model



#testcode
# from PIL import Image
# import time

# serve = modelserve()
# frame = Image.open(f'test.jpeg')
# # frame = Image.open(r"P:\Downloads\face.png")
# normalmodel = torch.load('model_mnist1.pickle', map_location=torch.device(serve.device))

# st = time.time()
# p = torch.argmax(normalmodel(transforms.ToTensor()(frame).unsqueeze(0)))
# normaltime = time.time() - st

# st = time.time()
# label = serve.predict(frame)
# qtime = time.time()-st

# print(p, label, normaltime, qtime)
        