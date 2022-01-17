# 🌟Mask Status Classification Service
데모 페이지: https://dc49-146-56-140-223.ngrok.io/  
2021.09.04 ~ 2021.10.27

## 📑Summary

> [부스트캠프 이미지 분류 대회](https://github.com/boostcampaitech2/image-classification-level1-12/tree/master)의 모델을 사용하여, 마스크 착용 상태 구분 웹서비스  
> 마스크 착용 상태, 연령대, 성별을 구분하고, 마스크를 올바르게 착용하지 않았을 시 경고메세지를 띄워줌  

## 👋 Team

|                            허정훈                            |                            임성민                            |                            이준혁                            |                            오주영                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/54921730?v=4)](https://github.com/herjh0405) | [![Avatar](https://avatars.githubusercontent.com/u/49228132?v=4)](https://github.com/mickeyshoes) | [![Avatar](https://avatars.githubusercontent.com/u/49234207?v=4)](https://github.com/kmouleejunhyuk) | [![Avatar](https://avatars.githubusercontent.com/u/69762559?v=4)](https://github.com/Jy0923) |
|                  `얼굴 인식 및 분류 모델링`                  |                     `모델 서빙` `백엔드`                     |                   `모델 경량화` `아이디어`  `프론트/백 총괄`                   |                         `프론트엔드`                         |

## 사용방법
![32](https://user-images.githubusercontent.com/49234207/138724656-beb67505-76c9-4bb6-85cf-f258c7cba173.png)
  

## 프로세스 flow
![process](https://user-images.githubusercontent.com/49234207/138725376-95a17c4e-d0d0-426e-bab0-fe7ce502f3ac.png)
*update(20211102): 링크에서는 이제 cropface 클래스를 사용해 얼굴이 없을 경우 NO Face 라벨을 출력합니다

### File Structure
```text
Mask_Status_Classification
├── README.md
├── future
│   └── cam.py          # use local cam
├── main.py             # execute main
├── model.onnx
├── model.pickle
├── requirements.txt
├── templates
│   └── index.html
└── utils
    └── modelserve.py   # quantize model
```

### Install Requirements
```
$ pip install -r requirements.txt
```

### Run server

```
$ python main.py
```
