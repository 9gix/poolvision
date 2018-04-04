import cv2
import numpy as np
from abc import ABCMeta

class Pocket(metaclass=ABCMeta):
    pass

class CornerPocket(Pocket):
    pass

class CenterPocket(Pocket):
    pass

class PoolTable(object):

    """
    Pocket Reference Numbering
    o---------o         p1: Top Left Pocket
    |         |         p2: Top Right Pocket
    |    o    |         p3: Left Center Pocket
    |         |         p4: Right Center Pocket  
    |         |         p5: Bottom Left Pocket
    o         o         p6: Bottom Right Pocket
    |         |         
    |    ^    | 
    |   < >   | 
    |    v    |
    o---------o
    """
    # List of Pocket object
    _pockets = [
        CornerPocket(),
        CornerPocket(),
        CenterPocket(),
        CenterPocket(),
        CornerPocket(),
        CornerPocket(),
    ]

    # Sets of Ball object
    _balls = {

    }

    cueball = None
    _size = None

    def __init__(self, size):
        """Size of the play area in metric, (length,width)"""
        self._size = size

class MagicRack(object):
    pass

class Ball(metaclass=ABCMeta):
    def __init__(self, number):
        self._number = number

class CueBall(Ball):
    def __init__(self):
        super().__init__(0)

class ObjectBall(Ball):
    def __init__(self, number):
        super().__init__(number)


class PoolVision(object):
    _table = None
    def __init__(self, cap):
        self.cap = cap
        
        if self.cap.isOpened():
            ret, self.raw_frame = self.cap.read()
            self.process_frame()
            cv2.imshow('Pool Vision',self.frame)

        self.cueball_track = None

    def detectCloth(self):
        self.blue = cv2.inRange(self.hsv, np.array([100, 60, 60]), np.array([110, 255, 255]))
        cloth = cv2.erode(self.blue, np.ones((5,5),np.uint8), iterations=3)
        cloth = cv2.dilate(self.blue, np.ones((5,5),np.uint8), iterations=14)
        cloth = cv2.bitwise_and(self.gray, self.gray, mask=cloth)
        self.cloth = cloth

    def detectTableManual(self):
        self.p1 = (235, 686)  # Pocket 1,2,5,6 (see ref above)
        self.p2 = (366, 530)
        self.p5 = (1210, 680)
        self.p6 = (1050, 530)

        self.p3 = ((self.p5[0] + self.p1[0]) / 2, (self.p5[1] + self.p1[1]) / 2)
        self.p4 = ((self.p6[0] + self.p2[0]) / 2, (self.p6[1] + self.p2[1]) / 2)

        self.table_mask = np.zeros((self.raw_frame.shape[:2]), np.uint8)
        self.table_poly = np.array([self.p1,self.p2,self.p6,self.p5])
        cv2.fillConvexPoly(self.table_mask, self.table_poly, 255)
        
    def detectTable(self):
        self.detectTableManual()
        self.detectCloth()
        self.gray = cv2.bitwise_and(self.gray, self.gray, mask=self.cloth)
        # TODO: Build a Table Model
        return PoolTable((2743,1372))

    def detectCueBall(self):
        # Find White Circle
        white_circle = cv2.inRange(self.hsv, np.array([0, 0, 110]), np.array([180, 30, 255]))
        whiteball_mask = cv2.bitwise_and(white_circle, white_circle, mask=self.table_mask)
        whiteball_mask = cv2.dilate(whiteball_mask, np.ones((5,5),np.uint8), iterations=2)
        whiteball_mask = cv2.erode(whiteball_mask, np.ones((5,5),np.uint8), iterations=4)
        whiteball_mask = cv2.dilate(whiteball_mask, np.ones((5,5),np.uint8), iterations=3)

        ctrs, hierarchy = cv2.findContours(whiteball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]
        cv2.drawContours(self.raw_frame, ctrs, -1, (0,255,0), 2)



        
        self.cueball_features = cv2.goodFeaturesToTrack(self.gray, maxCorners=5, qualityLevel=0.01, minDistance=5, mask=whiteball_mask)
        if self.cueball_features is not None:
            for f in self.cueball_features:
                x,y = f.ravel()
                cv2.circle(self.raw_frame, (x,y), 4, (0, 255, 255))

    def trackCueBall(self):
        #cv.calcOpticalFlowPyrLK(self.prev_gray, self.gray)
        pass

    def detectObjectBalls(self):
        # Find 9 colored Balls on the table exclude white.
        pass

    def detectBoundary(self):
        table_edge = cv2.bitwise_and(self.edge, self.edge, mask=self.table_mask)
        #lines = cv2.HoughLinesP(table_edge, rho=1, theta=np.pi/180, threshold=10, minLineLength=130, maxLineGap=10)

        # TODO: Need to determine the table region from the detected lines.
        return []


    def detectCircle(self):
        pocket_color = cv2.inRange(self.hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        table_mask = cv2.bitwise_and(self.table_mask, cv2.bitwise_not(pocket_color))
        table_mask = cv2.dilate(table_mask, np.ones((3,3), np.uint8), iterations=3)

        # Exclude All having blue cloth colors
        blue_cloth_mask = cv2.inRange(self.hsv, np.array([100, 10, 10]), np.array([110, 200, 255]))
        ball_mask = cv2.bitwise_not(blue_cloth_mask, mask=table_mask)

        # Exclude Pockets
        #pocket_color = cv2.inRange(self.hsv, np.array([0, 0, 200]), np.array([255, 255, 255]))

        # Remove Noise (white patches on the table cloth)
        ball_mask = cv2.erode(ball_mask, np.ones((5,5), np.uint8), iterations=3)
        ball_mask = cv2.dilate(ball_mask, np.ones((5,5), np.uint8), iterations=4)

        #cv2.imshow('xx', cv2.inRange(self.hsv, np.array([100, 0, 0]), np.array([105, 255, 255])))
        ball_candidate = cv2.bitwise_and(self.gray, self.gray, mask=ball_mask)
        cv2.imshow('xx', ball_candidate)



        #ctrs, hierarchy = cv2.findContours(table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]
        #cv2.drawContours(self.raw_frame, ctrs, -1, (255,255,0), 2)
        # self.ball_mask = cv2.inRange(self.hsv, np.array([100, 60, 60]), np.array([110, 255, 255]))
        # self.ball_mask = cv2.bitwise_and(ball_mask, ball_mask, mask=self.table_mask)
        # self.cloth = cv2.bitwise_and(self.gray, self.gray, mask=self.ball_mask)
        return cv2.HoughCircles(ball_candidate, method=cv2.HOUGH_GRADIENT, dp=1, minDist=14, param1=300, param2=4, minRadius=14, maxRadius=15)[0][:10]

    def overlayCircle(self, circle):
        cv2.circle(self.raw_frame, (circle[0], circle[1]), circle[2], (0, 255,0))
    
    def overlayLine(self, x1, y1, x2, y2):
        cv2.line(self.raw_frame, (x1,y1),(x2,y2),(0,255,0),2)

    def process_frame(self):
        # TODO: Frame Preparation
        try:
            self.prev_frame = self.frame
            self.prev_gray = self.gray
        except AttributeError:
            pass
        self.frame = self.raw_frame.copy()
        self.frame = cv2.GaussianBlur(self.frame, (5,5), 0)
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.edge = cv2.Canny(self.frame, 25, 50)

        # TODO: Table Detection
        if not self._table:
            self._table = self.detectTable()

        # TODO: Ball Detection
        self.detectCueBall()
        #self.detectObjectBalls()

        # TODO: Ball tracking
        self.trackCueBall()

        # TODO: Player Detection

        # TODO: Player Tracking

        # TODO: Homography Transformation to Top Down View

        # TODO: Further Analysis

        # TODO: Video Presentation

        circles = self.detectCircle()
        for circle in circles:
            self.overlayCircle(circle)


    def run(self):       
        while(self.cap.isOpened()):
            ret, self.raw_frame = self.cap.read()
            self.process_frame()
            cv2.imshow('Pool Vision',self.raw_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    v1 = 'vids/1.mp4'
    cap = cv2.VideoCapture(v1)
    pool_vision = PoolVision(cap)
    cv2.namedWindow('Pool Vision', cv2.WINDOW_NORMAL)
    pool_vision.run()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()