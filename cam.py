import albumentations
import albumentations.pytorch
import cv2
import cvlib as cv
import torch
import torch.nn.functional as F

model = torch.load("model.pickle", map_location=torch.device("cpu"))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 내장 카메라를 받아옴 - 번호(0)
capture = cv2.VideoCapture(0)
# cap.set(propid, value) - propid:속성, value:값
# 카메라의 너비를 640, 높이를 480으로 변경
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

transform = albumentations.Compose(
    [
        albumentations.Resize(512, 384, cv2.INTER_LINEAR),
        albumentations.Normalize(mean=(0.5), std=(0.2)),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)


# padding value before cropping
X_PADDING = 20
Y_PADDING = 30  # gave more padding due to include mask on the chin & hair sty

# cv2.waitKey(delay) - delay시간만큼 키 입력을 기다림
while cv2.waitKey(100) != ord("q"):
    # check - 카메라의 상태, 정상 작동:True
    # frame - 현재 시점의 프레임이 저장
    check, frame = capture.read()
    # cv2.imshow(winname, mat) 창의 이름과 이미지를 할당
    H, W, C = frame.shape
    result_detected = cv.detect_face(frame)
    if len(result_detected)!=0:
        try :
            if type(result_detected[1][0]) == list:
                print(type(result_detected[1][0]))
                prob = result_detected[1][0][0]
            else:
                prob = result_detected[1][0]
            if prob > 0.8:
                xmin = max(int(result_detected[0][0][0]) - X_PADDING, 0)
                ymin = max(int(result_detected[0][0][1]) - Y_PADDING, 0)
                xmax = min(int(result_detected[0][0][2]) + X_PADDING, 640)
                ymax = min(int(result_detected[0][0][3]) + Y_PADDING, 480)

                bbox = cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2
                )
                image_array = frame[ymin:ymax, xmin:xmax]
                augmented = transform(image=image_array)
                image = augmented["image"]
                image = image.unsqueeze(dim=0).to(device)
                with torch.no_grad():
                    pred = model(image)
                    pred = F.softmax(pred, dim=1).cpu().numpy()
                    pred = pred.argmax()
                    cv2.putText(frame,str(pred.item()),(xmax, ymax),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,
                    )
        except :
            pass
    cv2.imshow("video", frame)

capture.release()
cv2.destroyAllWindows()
