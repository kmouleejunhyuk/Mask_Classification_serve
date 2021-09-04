#py for retriving camera frames & motions(정훈)
#input: 프로그램 시작 여부(불분명, 자유)
#output: face frame(img형태, 크기 고정 & 명시 부탁)
#optional: 디버깅 위한 화면 표현(face detected, label같은 문자)
import cv2
import numpy as np
 


class cam():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        self.thresh = 25
        self.max_diff = 5
        self.f1, self.f2, self.f3 = None, None, None


    def cam_main(self):
        if self.cap.isOpened():
            ret, self.f1 = self.getframe()
            ret, self.f2 = self.getframe()

            while ret:
                ret, self.f3 = self.getframe()
                ret, diff, diff_cnt = self.getdiff()
                draw = self.f3.copy()

                if diff_cnt > self.max_diff:
                    #dl part
                    draw = self.contour(diff, draw)

                cv2.imshow('motion', draw)

                self.f1, self.f2 = self.f2, self.f3

                if cv2.waitKey(33) > 0:
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        return None

    def getdiff(self):
        '''
        get diff of frame1, frame2, frame3
        for motion detecting
        '''

        grays = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in [self.f1, self.f2, self.f3]]
        diffs = [cv2.absdiff(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]
        ret_diff = [cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY) for diff in diffs]
        
        ret = all([x[0] for x in ret_diff])
        diff_threshs = [x[1] for x in ret_diff]
 
        diff = cv2.bitwise_and(diff_threshs[0], diff_threshs[1])

        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

        diff_cnt = cv2.countNonZero(diff)

        return ret, diff, diff_cnt



    def getframe(self):
        '''
        equivalant to cv2 capture read
        '''
        ret, frame = self.cap.read()
        return ret, frame


    def contour(self, diff, draw):
        '''
        give contour to frame
        '''
        nzero = np.nonzero(diff)
        cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                        (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)
        cv2.putText(draw, "Motion detected!!", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        return draw
