#py for model lightweighting & inferencing
#input: img(facecropped)
#output: label, alert/pass 여부
from typing import Tuple
import torch
import os
import onnx
from onnx_tf.backend import prepare


class modelserve():
    def __init__(self, quantize: str = 'bfloat16', model_dir: str = r'./model.pickle', imagesize: Tuple=(1, 3, 584, 312)):
        self.modeldir = model_dir
        self.quantize = quantize
        self.imagesize = imagesize
        self.device = "cpu"
        self.to_tf(self.load_model())
        


    def trans_image(self, image):
        img = image.unsqueeze(0)
        return img


    def to_tf(self, model):
        model.eval()
        # 모델에 대한 입력값
        x = torch.randn(1, 3, 512//4, 384//4, requires_grad=True)
        torch_out = model(x)

        # 모델 변환
        torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "model.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

        onnx_model = onnx.load("output/model.onnx")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph("output/model.pb")  
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
        