import numpy as np
from face_landmarks import convert_landmarks, get_landmark_index_dict


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
        self.landmark_type = landmark_type
        self.landmark_index_dict = get_landmark_index_dict(landmark_type)
        return

    def get_bounding_box(self, part = 'face'):
        # print('part is {}'.format(part))
        # print('landmarks are {}'.format(self.landmarks))
        # print('landmarks index dict are {}'.format(self.landmark_index_dict[part]))
        landmarks = np.array([pt for pt in self.landmarks[self.landmark_index_dict[part]] if (pt-np.array([-1, -1])).any()])
        x2, y2 = np.amax(landmarks, axis = 0)
        x1, y1 = np.amin(landmarks, axis = 0)
        assert x1 >=0 and y1 >=0
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
        assert(self.landmarks[landmark_a][0] >= 0 and self.landmarks[landmark_a][1] >= 0 \
               and self.landmarks[landmark_b][0] >= 0 and self.landmarks[landmark_b][1] >= 0)
        return np.linalg.norm(self.landmarks[landmark_a] - self.landmarks[landmark_b])

    def convert_to_landmark_type(self, landmark_type):
        if landmark_type == self.landmark_type:
            # print('landmark is already type {}'.format(landmark_type))
            return
        else:
            self.landmarks = convert_landmarks(self.landmarks, self.landmark_type, landmark_type)
            self.landmarks_homogenous = np.insert(self.landmarks, 2, 1.0, axis = 1)
            self.landmark_type = landmark_type
            self.landmark_index_dict = get_landmark_index_dict(landmark_type)

class FaceData:
    def __init__(self):
        self.items = []
        self.X = []
        self.y = {}

    def append(self, face_landmark):
        assert isinstance(face_landmark, FaceLandmark)
