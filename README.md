# 🌟Mask Status Classification Service

2021.09.04 ~

## 📑Summary

> [부스트캠프 이미지 분류 대회](https://github.com/boostcampaitech2/image-classification-level1-12/tree/server-hun)의 모델을 사용하여, 마스크 착용 상태 구분 웹서비스  
> 마스크 착용 상태, 연령대, 성별을 구분하고, 마스크를 올바르게 착용하지 않았을 시 경고메세지를 띄워줌  

## 👋 Team

|                            허정훈                            |                            임성민                            |                            이준혁                            |                            오주영                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/54921730?v=4)](https://github.com/herjh0405) | [![Avatar](https://avatars.githubusercontent.com/u/49228132?v=4)](https://github.com/mickeyshoes) | [![Avatar](https://avatars.githubusercontent.com/u/49234207?v=4)](https://github.com/kmouleejunhyuk) | [![Avatar](https://avatars.githubusercontent.com/u/69762559?v=4)](https://github.com/Jy0923) |
|                  `얼굴 인식 및 분류 모델링`                  |                     `모델 서빙` `백엔드`                     |                   `모델 경량화` `아이디어`                   |                         `프론트엔드`                         |

## 📅History

|    Date    | Contributor | Descrption                                                   |
| :--------: | :---------: | :----------------------------------------------------------- |
| 2021.09.04 |    모두     | 팀 결성, 역할 분담, Organization 생성, Github과 Asana를 기반으로 협업 |
|            |             |                                                              |

### File Structure
```text
Mask_Status_Classification
├── README.md
├── __pycache__
│   └── cam.cpython-37.pyc
├── backend
│   ├── backend
│   ├── db.sqlite3
│   ├── manage.py
│   ├── predict
│   └── webcam_test
├── cam.py
├── feedback.py
├── main.py
├── model_mnist1.pickle
├── modelserve.py
├── requirements.txt
├── test.jpeg
├── test.py
```

### Install Requirements
```
$ pip install -r requirements.txt
```

### run server

```
$ python manage.py runserver
```