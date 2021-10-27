#py for model lightweighting & inferencing
#input: img(facecropped)
#output: label, alert/pass 여부
#ONNX 변환으로 인해 전체 비활성화
#ONNX 변환이 필요할 떄만 일시적으로 활성화할 것
#pip install torch

from typing import Tuple
import torch
import os

'''
class for testing & changing model weight files(.pth)

usage example
serve = modelserve(prefered args)
'''
class modelserve():
    def __init__(self, quantize: str = 'normal', model_dir: str = r'./model.pickle', imagesize: Tuple=(1, 3, 512, 384)):
        self.modeldir = model_dir
        self.quantize = quantize
        self.imagesize = imagesize
        self.device = "cpu"
        self.torchmodel = self.load_model()
        self.to_tf(self.torchmodel)
        

    def trans_image(self, image):
        return image.unsqueeze(0)


    def to_ONNX(self, model):
        model.eval()
        x = torch.randn(1, 3, 512, 384, requires_grad=True)

        torch.onnx.export(model, x, "model.onnx", export_params=True, do_constant_folding=True,
                  input_names = ['input'], output_names = ['output'])

        print('complete')
        return None


    def predict(self, image):
        with torch.no_grad():
            onehot_label = self.model(image.unsqueeze(0))
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
            return torch.quantization.quantize_dynamic(model, dtype=torch.qint8)  #quantizing model is in beta. may not work properly

        elif self.quantize == 'bfloat16':
            for layer in model.parameters():
                if not isinstance(layer, torch.nn.BatchNorm2d):
                    layer.bfloat16()
            return model

        else:
            return model



