from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self, extraction_type="mixed"):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.faces = None
        self.landmarks = None
        self.calibration = Calibration()
        self.count_cnn = 0
        self.count_cnn_success = 0
        self.extraction_type = extraction_type
        self._face_detector = dlib.get_frontal_face_detector()
        self._face_detector_cnn = dlib.cnn_face_detection_model_v1('trained_models/mmod_human_face_detector.dat')

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    @property
    def face_located(self):
        """Check that the pupils have been located"""
        try:
            test = self.faces.left()
            return True
        except Exception:
            return False
    @property
    def landmarks_located(self):
        """Check that the pupils have been located"""
        try:
            test = self.landmarks.parts()
            return True
        except Exception:
            return False

    def store_image(self,frame, d, export_path=None):
        if export_path is not None:
            b = frame[d.top():d.bottom(),d.left():d.right()]
            cv2.imwrite(export_path, b)

    def _analyze(self, export_path=None):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        try:
            if self.extraction_type == 'mixed':
                self.faces = self._face_detector(frame)[0]
                self.landmarks = self._predictor(frame, self.faces)
                self.eye_left = Eye(frame, self.landmarks, 0, self.calibration)
                self.eye_right = Eye(frame, self.landmarks, 1, self.calibration)
                self.store_image(self.frame, self.faces, export_path)
            else:
                raise IndexError
        except IndexError:
            try:
                self.count_cnn = self.count_cnn + 1
                self.faces = self._face_detector_cnn(frame)[0].rect               
                self.landmarks = self._predictor(frame, self.faces)
                self.eye_left = Eye(frame, self.landmarks, 0, self.calibration)
                self.eye_right = Eye(frame, self.landmarks, 1, self.calibration)

                self.store_image(self.frame, self.faces, export_path)
                self.count_cnn_success = self.count_cnn_success + 1
            except IndexError:
                print(' - failed')
                self.faces = None
                self.landmarks = None
                self.eye_left = None
                self.eye_right = None


    def refresh(self, frame,export_path=None):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze(export_path)

    def close_run(self):
        print("Tried CNN for faces ", self.count_cnn, " with sucess ", self.count_cnn_success)

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    def features(self):
        #print(self.eye_left.blinking if self.horizontal_ratio() is not None else 0, 4)
        
        def check_blinking(eye):

            if self.eye_left.blinking is None:
                return 0
            else:
                return self.eye_left.blinking

        features = {
            'pupils_located': [1]  if self.pupils_located else [0]
            , 'left_coords': list(self.pupil_left_coords()) if self.pupils_located else [0, 0]
            , 'right_coords': list(self.pupil_right_coords()) if self.pupils_located else [0, 0]
            , 'eye_left': [self.eye_left.pupil.y  if self.pupils_located else 0
                            ,self.eye_left.pupil.x  if self.pupils_located else 0
                            ,self.eye_left.center[0]  if self.pupils_located else 0
                            ,self.eye_left.center[1]  if self.pupils_located else 0
                            ,self.eye_left.origin[0] if self.pupils_located else 0
                            ,self.eye_left.origin[1] if self.pupils_located else 0]
            , 'eye_right': [self.eye_right.pupil.y if self.pupils_located else 0
                            ,self.eye_right.pupil.x if self.pupils_located else 0
                            ,self.eye_right.center[0] if self.pupils_located else 0
                            ,self.eye_right.center[1] if self.pupils_located else 0
                            ,self.eye_right.origin[0] if self.pupils_located else 0
                            ,self.eye_right.origin[1] if self.pupils_located else 0]
            , 'faces': [self.faces.left() if self.face_located else 0
                        ,self.faces.top() if self.face_located else 0
                        ,self.faces.right() if self.face_located else 0
                        ,self.faces.bottom() if self.face_located else 0
                        ,self.faces.area() if self.face_located else 0
                        ,self.faces.height()  if self.face_located else 0
                        ,self.faces.width() if self.face_located else 0]
            , 'landmarks': [part for part in self.landmarks.parts()] if self.landmarks_located else [0 for i in range(136)] 
            #, 'is_left':self.is_left() if self.is_left()  is not None else 0
            #, 'is_right':self.is_right()
            , 'horizontal_ratio': [round(self.horizontal_ratio() if self.horizontal_ratio() is not None else 0, 4)]
            , 'vertical_ratio': [round(self.vertical_ratio() if self.vertical_ratio() is not None else 0, 4)]
            #, 'is_center': self.is_center()
            #, 'is_blinking': self.is_blinking()
            , 'left_blinking_rate' : [round(check_blinking(self.eye_left) if self.horizontal_ratio() is not None else 0, 4)]
            , 'right_blinking_rate' : [round(check_blinking(self.eye_right)  if self.horizontal_ratio() is not None else 0, 4)]
        }

        if self.landmarks_located:
            landsmarks_flat = []
            for part in features['landmarks']:
                landsmarks_flat.append(part.x)
                landsmarks_flat.append(part.y)
            features['landmarks'] = landsmarks_flat

        feature_vector = [v for k,v in features.items()] #TODO: Save to be on the save side
        
        feature_vector = [item for sublist in feature_vector for item in sublist]
        feature_vector = [i if i is not None else 0 for i in feature_vector]

        return features, feature_vector
