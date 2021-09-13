#py for model lightweighting & inferencing
#do not use this script in serving level
import io
from typing import Tuple
import os
import numpy as np
from torchvision.transforms.transforms import CenterCrop, RandomHorizontalFlip
import torchvision.transforms as transforms
import torch

class modelserve():
    
    def __init__(self, quantize: str = 'bfloat16', model_dir: str = r'./model.pickle', imagesize: Tuple=(1, 3, 584, 312)):
        self.modeldir = model_dir
        self.quantize = quantize
        self.mean = (0.5601, 0.5241, 0.5014)
        self.std = (0.2331, 0.2430, 0.2456)
        self.imagesize = imagesize
        self.device = "cpu"
        self.model = self.load_model()


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
                self.to_jit(model)
                return None

            except:
                raise Exception('model save failed')


    def to_jit(self, model):
        '''
        if not loading jit, save as jit and reload model as jit
        '''
        model.eval()
        dummy_input = torch.randn(1, 3, 584, 312)
        torch.onnx.export(model, dummy_input, "Model.onnx", verbose=True)
        return None


modelserve()
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
        