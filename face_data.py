import numpy as np

class FaceLandmark:
    def __init__(self, image, landmarks, landmark_type):
        self.image = image
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        if len(image.shape) < 3:
            self.channels = 1
        else:
            self.channels = image.shape[2]
        self.landmarks = np.array(landmarks)
        self.landmarks_homogenous = np.insert(self.landmarks, 2, 1.0, axis = 1)
        self.landmarks_index_dict = {}
        self.landmark_type = landmark_type
        if landmark_type  == 'muct_clmtools':
            self.landmarks_index_dict['face'] = range(71)
            self.landmarks_index_dict['left_eye']  = [23, 24, 25, 26, 63, 64, 65, 66]
            self.landmarks_index_dict['right_eye'] = [28, 29, 30, 31, 67, 68, 69, 70]
            self.landmarks_index_dict['mouth'] = range(44, 62)
        else:
            assert False, '{} not supported'.format(landmark_type)
        return

    def get_bounding_box(self, part = 'face'):
        x2, y2 = np.amax(self.landmarks[self.landmarks_index_dict[part]], axis = 0)
        x1, y1 = np.amin(self.landmarks[self.landmarks_index_dict[part]], axis = 0)
        x1 = min(int(x1 + 0.5), self.cols - 1)
        x2 = min(int(x2 + 0.5), self.cols - 1)
        # extend further to forehead
        if part == 'face':
            y1 = max(int(y1 - (y2 - y1) * 0.2 + 0.5), 0)
        else:
            y1 = min(int(y1 + 0.5), self.rows - 1)
        y2 = min(int(y2 + 0.5), self.rows - 1)
        return [x1, y1, x2, y2]

    def get_distance(self, landmark_a, landmark_b):
        return np.linalg.norm(self.landmarks[landmark_a] - self.landmarks[landmark_b])

class FaceData:
    def __init__(self):
        self.items = []
        self.X = []
        self.y = {}

    def append(self, face_landmark):
        assert isinstance(face_landmark, FaceLandmark)
