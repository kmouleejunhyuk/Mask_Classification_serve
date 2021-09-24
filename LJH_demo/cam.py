import cv2
import cvlib as cv

class facecrop():
    #jung hun
    def __init__(self) -> None:
        self.PADDING = 100

    def cropface(self, frame) -> None:
        brgframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result_detected = cv.detect_face(brgframe)
        if len(result_detected)!=0:
            try:
                if type(result_detected[1][0]) == list:
                    print(type(result_detected[1][0]))
                    prob = result_detected[1][0][0]
                else:
                    prob = result_detected[1][0]

                if prob > 0.8:
                    xmin = max(int(result_detected[0][0][0]) - self.PADDING, 0)
                    ymin = max(int(result_detected[0][0][1]) - self.PADDING, 0)
                    xmax = min(int(result_detected[0][0][2]) + self.PADDING, 480)
                    ymax = min(int(result_detected[0][0][3]) + self.PADDING, 640)

                    image = frame[ymin:ymax, xmin:xmax]

                    return image
                else:
                    return 'NO FACE'

            except:
                return 'NO FACE'

        else:
            return 'NO FACE'