import cv2
import numpy as np
from matplotlib import pyplot as plt
import collections
from abc import ABCMeta
import time



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
        self.is_mouse_down = False
        self.selectKey = None
        self.selected_object_balls = {}
        self.cueball_track = collections.deque(maxlen=30)
        self.origin = (20, 20)
        self.width = 300

        if self.cap.isOpened():
            ret, self.raw_frame = self.cap.read()
            self.detectTableROI()
            self.process_frame()
            cv2.imshow('Pool Vision', self.frame)

    def detectCloth(self):
        self.blue = cv2.inRange(self.hsv, np.array([100, 60, 60]), np.array([110, 255, 255]))
        cloth = cv2.erode(self.blue, np.ones((5, 5), np.uint8), iterations=3)
        cloth = cv2.dilate(self.blue, np.ones((5, 5), np.uint8), iterations=14)
        cloth = cv2.bitwise_and(self.gray, self.gray, mask=cloth)
        self.cloth = cloth

    def detectTableROI(self):
        # self.table_ROI = self.raw_frame[530:686, 235:1210]

        # Manual hardcoded (Temporary for development purposes)
        self.p1 = (235, 686)  # Pocket 1,2,5,6 (see ref above)
        self.p2 = (366, 530)
        self.p5 = (1210, 680)
        self.p6 = (1050, 530)

        ctrs = np.array([np.array([self.p1, self.p2, self.p5, self.p6])])
        x,y,w,h = cv2.boundingRect(ctrs)
        self.left_offset,self.right_offset,self.top_offset,self.bottom_offset = (x,x+w,y,y+h)
        self.table_ROI = self.raw_frame[self.top_offset:self.bottom_offset,self.left_offset:self.right_offset]

        self.pt1 = (self.p1[0]-self.left_offset, self.p1[1] - self.top_offset)
        self.pt2 = (self.p2[0]-self.left_offset, self.p2[1] - self.top_offset)
        self.pt5 = (self.p5[0]-self.left_offset, self.p5[1] - self.top_offset)
        self.pt6 = (self.p6[0]-self.left_offset, self.p6[1] - self.top_offset)

        self.p3 = ((self.p5[0] + self.p1[0]) / 2, (self.p5[1] + self.p1[1]) / 2)
        self.p4 = ((self.p6[0] + self.p2[0]) / 2, (self.p6[1] + self.p2[1]) / 2)

        self.table_mask = np.zeros((self.table_ROI.shape[:2]), np.uint8)
        self.table_poly = np.array([self.pt1, self.pt2, self.pt6, self.pt5])
        cv2.fillConvexPoly(self.table_mask, self.table_poly, 255)
        cv2.imshow('table_mask', self.table_mask)

    def detectTable(self):
        self.detectCloth()
        self.gray = cv2.bitwise_and(self.gray, self.gray, mask=self.cloth)
        src = np.float32([self.p2, self.p6, self.p5, self.p1])
        dst = np.float32([self.origin, (self.origin[0] + 2 * self.width, self.origin[1]),
                          (self.origin[0] + 2 * self.width, self.origin[1] + self.width),
                          (self.origin[0], self.origin[1] + self.width)])
        
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        # TODO: Build a Table Model
        return PoolTable((2743, 1372))

    def detectCueBall(self):
        # Find White Circle
        s = time.process_time()
        white_circle = cv2.inRange(self.hsv, np.array([0, 0, 110]), np.array([180, 60, 255]))
        whiteball_mask = cv2.bitwise_and(white_circle, white_circle, mask=self.table_mask)
        cv2.imshow('whiteball_mask', self.table_mask)
        whiteball_mask = cv2.erode(whiteball_mask, np.ones((5, 5), np.uint8), iterations=2)
        whiteball_mask = cv2.dilate(whiteball_mask, np.ones((5, 5), np.uint8), iterations=2)
        ctrs, hierarchy = cv2.findContours(whiteball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]
        cv2.imshow('ctr', whiteball_mask)
        cv2.drawContours(self.frame, ctrs, -1, (255, 0, 255), 3)
        cv2.imshow('frm', self.frame)
        circles = list(filter(lambda circ: circ[1] > 13 and circ[1] < 20, list(map(lambda x: cv2.minEnclosingCircle(x), ctrs))))
        circles.sort(key=lambda circ: circ[1], reverse=True)
        if circles:
            largest_circle = circles[0]
            (x,y), r = largest_circle
            x+=self.left_offset
            y+=self.top_offset
            self.cueball = (int(x), int(y))
            self.cueball_track.appendleft(self.cueball)
            cv2.circle(self.raw_frame, (int(x), int(y)), int(r), (255, 0, 0), 2)
            pt = self.transformToTopDownView(x,y)
            cv2.circle(self.raw_frame, (int(pt[0]), int(pt[1])), int(r), (255, 0, 0), 2)

        
    def trackCueBall(self):
        # cv.calcOpticalFlowPyrLK(self.prev_gray, self.gray)
        cueball_track = iter(self.cueball_track)
        prev_cueball = next(cueball_track)
        for curr_cueball in cueball_track:
            x = np.linalg.norm(np.array(curr_cueball) - np.array(prev_cueball))
            if x < 50:
                cv2.line(self.raw_frame, prev_cueball, curr_cueball, (255, 0, 0), 2)
            a = self.transformToTopDownView(*np.float32(prev_cueball))
            b = self.transformToTopDownView(*np.float32(curr_cueball))
            if x < 50:
                cv2.line(self.raw_frame, tuple(a), tuple(b), (255, 0, 0), 2)
            prev_cueball = curr_cueball

    def detectBoundary(self):
        # table_edge = cv2.bitwise_and(self.edge, self.edge, mask=self.table_mask)
        # lines = cv2.HoughLinesP(table_edge, rho=1, theta=np.pi/180, threshold=10, minLineLength=130, maxLineGap=10)

        # TODO: Need to determine the table region from the detected lines.
        return []

    def detectCircle(self):
        u = time.process_time()
        blue_cloth_mask = cv2.inRange(self.hsv, np.array([100, 10, 10]), np.array([110, 200, 255]))
        blue_cloth_mask = cv2.bitwise_and(blue_cloth_mask, blue_cloth_mask, mask=self.table_mask)
        blue_cloth_mask = cv2.dilate(blue_cloth_mask, np.ones((5, 5), np.uint8), iterations=4)
        blue_cloth_mask = cv2.erode(blue_cloth_mask, np.ones((5, 5), np.uint8), iterations=4)
        # table_mask = cv2.bitwise_and(self.table_mask, blue_cloth_mask)
        # ctrs, hier = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]
        # cv2.drawContours(table_mask, ctrs, 0, 255, -1)

        # Exclude All having blue cloth colors
        ball_mask = cv2.bitwise_not(blue_cloth_mask, mask=self.table_mask)

        # Remove Noise (white patches on the table cloth)
        ball_mask = cv2.erode(ball_mask, np.ones((5, 5), np.uint8), iterations=2)
        ball_mask = cv2.dilate(ball_mask, np.ones((5, 5), np.uint8), iterations=10)
        # cv2.imshow('xx', cv2.inRange(self.hsv, np.array([100, 0, 0]), np.array([105, 255, 255])))
        ball_candidate = cv2.bitwise_and(self.gray, self.gray, mask=ball_mask)
        # cv2.imshow('xx', ball_candidate)
        # ctrs, hierarchy = cv2.findContours(table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]
        # cv2.drawContours(self.raw_frame, ctrs, -1, (255,255,0), 2)
        # self.ball_mask = cv2.inRange(self.hsv, np.array([100, 60, 60]), np.array([110, 255, 255]))
        # self.ball_mask = cv2.bitwise_and(ball_mask, ball_mask, mask=self.table_mask)
        # self.cloth = cv2.bitwise_and(self.gray, self.gray, mask=self.ball_mask)
        circles = cv2.HoughCircles(ball_candidate, method=cv2.HOUGH_GRADIENT, dp=1, minDist=14, 
            param1=300, param2=7, minRadius=14, maxRadius=15)
        cv2.imshow('bc', ball_candidate)
        if circles is not None:
            for circle in circles[0]:
                circle[0] += self.left_offset
                circle[1] += self.top_offset
                self.overlayCircle(circle)
                center = self.transformToTopDownView(circle[0], circle[1])
                circle = (center[0], center[1], circle[2])
                self.overlayCircle(circle)

    def transformToTopDownView(self, x,y):
        return cv2.perspectiveTransform(np.array([[[x, y]]]), self.matrix)[0][0]
    
    def overlayCircle(self, circle):
        cv2.circle(self.raw_frame, (circle[0], circle[1]), circle[2], (0, 0, 255))

    def overlayLine(self, x1, y1, x2, y2):
        cv2.line(self.raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def drawTable(self):
        cv2.circle(self.raw_frame, self.origin, 15, (0, 255, 0))
        cv2.circle(self.raw_frame, (self.origin[0] + self.width, self.origin[1]), 15, (0, 255, 0))
        cv2.circle(self.raw_frame, (self.origin[0] + 2 * self.width, self.origin[1]), 15, (0, 255, 0))
        cv2.circle(self.raw_frame, (self.origin[0], self.origin[1] + self.width), 15, (0, 255, 0))
        cv2.circle(self.raw_frame, (self.origin[0] + self.width, self.origin[1] + self.width), 15, (0, 255, 0))
        cv2.circle(self.raw_frame, (self.origin[0] + 2 * self.width, self.origin[1] + self.width), 15, (0, 255, 0))
        cv2.rectangle(self.raw_frame, (self.origin[0], self.origin[1]),
                      (self.origin[0] + self.width * 2, self.origin[1] + self.width), (0, 255, 0), 3)

    def process_frame(self):
        # Process Table ROI (including balls)
        self.detectTableROI()
        self.process_table_roi()

        # Process Player ROI
        self.process_player_roi()

    def process_player_roi(self):
        pass

    def process_table_roi(self):
        t = time.process_time()
        # TODO: Frame Preparation
        try:
            self.prev_frame = self.frame
            self.prev_gray = self.gray
        except AttributeError:
            pass
        self.frame = self.table_ROI.copy()

        self.frame = cv2.GaussianBlur(self.frame, (5, 5), 0)
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # self.edge = cv2.Canny(self.frame, 25, 50)

        # TODO: Table Detection
        self._table = self.detectTable()

        self.drawTable()

        cv2.imshow('table', self.gray)
        # TODO: Ball Detection
        self.detectCueBall()
        self.detectObjectBalls()
        # TODO: Ball tracking
        self.trackCueBall()

        self.trackObjectBalls()
        self.renderObjectBalls()
        # TODO: Player Detection

        # TODO: Player Tracking

        # TODO: Homography Transformation to Top Down View


        # TODO: Further Analysis

        # TODO: Video Presentation

    def detectObjectBalls(self):
        # Find 9 colored Balls on the table exclude white.

        # circles = self.detectCircle()
        # for ball_descriptor in ball_descriptors:
        #     cv2.

        for no, ball in self.selected_object_balls.items():
            if ball.get('is_detected'):
                continue

            # Identify Color Space within the circle.
            (x,y), r = ball['pts'], ball['r']
            ball['frm'] = self.raw_frame[y-r:y+r, x-r:x+r]
            cv2.imshow(str(no), ball['frm'])
            # circles = cv2.HoughCircles(ball['frm'], method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
            #     param1=50, param2=30, minRadius=0, maxRadius=0)
            # if circles is not None:
            #     circle_mask = np.zeros((ball['frm'].shape[:2]), np.uint8)
            #     cv2.circle(circle_mask, circles[0], circles[1], 255, -1)

            # cv2.imshow('xxx', circle_mask)
            # Channel Split
            ball['hsv'] = cv2.cvtColor(ball['frm'], cv2.COLOR_BGR2HSV)
            ball['mask'] = cv2.bitwise_not(cv2.inRange(ball['hsv'], np.array([100, 60, 60]), np.array([110, 255, 255])))
            cv2.imshow('ball_mask', ball['mask'])
            ball['hist'] = cv2.calcHist([ball['hsv']], channels=[0], mask=ball['mask'], histSize=[180], ranges=[0, 180])
            ball['is_detected'] = True
        # pass
        # Distinguish each color ball and find a good corner in each of the ball
            cv2.normalize(ball['hist'], ball['hist'], 0, 255, cv2.NORM_MINMAX)
            ball['track_window'] = (
                ball['pts'][0] - ball['r'] - self.left_offset, 
                ball['pts'][1] - ball['r'] - self.top_offset, 
                ball['d'], 
                ball['d']
            )

    def trackObjectBalls(self):
        for no, ball in self.selected_object_balls.items():
            # Identify Color Space within the circle.
            ball['back-projection'] = cv2.calcBackProject([self.hsv], channels=[0], hist=ball['hist'], ranges=[0, 180], scale=1)
            cv2.imshow('bp', ball['back-projection'])
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
            ret, ball['track_window'] = cv2.meanShift(ball['back-projection'], ball['track_window'], term_crit)

    def renderObjectBalls(self):
        for no, ball in self.selected_object_balls.items():
            if ball.get('track_window'):
                x,y,w,h = ball['track_window']
                cv2.rectangle(self.raw_frame, (x,y), (x+w,y+h), 255,2)
                r = w//2
                pts_offset = (self.left_offset + x + r , self.top_offset + y + r)
                cv2.circle(self.raw_frame, pts_offset, r, (0, 0, 255), 2)


    def mouseEvent(self, event, x, y, flags, param):
        x,y = int(x),int(y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_mouse_down = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_mouse_down == True:
                if type(self.selectKey) is int:
                    # Selecting Ball
                    self.ir = int(np.linalg.norm(np.array([x,y]) - np.array([self.ix, self.iy])))
                    tmp_frm = self.raw_frame.copy()
                    cv2.circle(tmp_frm,(self.ix,self.iy),self.ir,(0,255,0),1)
                    cv2.imshow('Pool Vision', tmp_frm)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_mouse_down = False
            if self.selectKey:
                self.selected_object_balls[self.selectKey] = {
                    'pts': (self.ix, self.iy),
                    'r': self.ir,
                    'd': 2*self.ir
                }
                self.renderObjectBalls()


    def run(self):
        cv2.setMouseCallback('Pool Vision', self.mouseEvent)
        while (self.cap.isOpened()):
            ret, self.raw_frame = self.cap.read()

            self.process_frame()
            cv2.imshow('Pool Vision', self.raw_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                cv2.waitKey(0)
            elif k == ord('.'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 300)
            elif k in [ord(str(i)) for i in range(0,10)]:
                self.selectKey = int(chr(k))
                cv2.waitKey(0)
                self.selectKey = None

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
