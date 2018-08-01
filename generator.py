import numpy as np
import random
import cv2
from keras.preprocessing.image import Iterator

from face_landmarks import flip_landmarks
from face_data import FaceLandmark

class LandmarkDictionaryIterator(Iterator):
    def __init__(self, face_landmark_list, face_landmark_indices,
                 image_face_landmark_data_generator,
                 n, shuffle, random_horizontal_flip,
                 target_size = (64, 64), batch_size=32, seed=None):
        self.face_landmark_list = face_landmark_list
        self.face_landmark_indices = face_landmark_indices
        self.generator = image_face_landmark_data_generator
        self.shuffle = shuffle
        self.random_horizontal_flip = random_horizontal_flip
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed
        super(LandmarkDictionaryIterator, self).__init__(n, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        X = []
        y = []
        for index in index_array:
            fl = self.face_landmark_list[index]
            if fl.image is None:
                image = cv2.imread(fl.get_image_path())
            else:
                image = fl.image
            if self.random_horizontal_flip:
                flip = random.choice([True, False])
                if flip:
                    image = np.fliplr(image)
                    landmarks = (np.array([image.shape[1]-1, 0]) - fl.landmarks) * np.array([1, -1])
                    flip_landmarks(landmarks, fl.landmark_type)
                    fl = FaceLandmark(landmarks, fl.landmark_type, image)

            # rotation param: rotation center and angle
            x1, y1, x2, y2 = fl.get_bounding_box(self.generator.rotate_bounding_box_part)
            rotate_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            angle = (random.random()*2-1) * self.generator.rotate_limit_in_degrees
            # scale param: scale ratio
            x1, y1, x2, y2 = fl.get_bounding_box(self.generator.scale_bounding_box_part)
            scale_ratio = min(self.generator.scale_bounding_box_size/1.0/(y2-y1),
                              self.generator.scale_bounding_box_size/1.0/(x2-x1))
            ratio = ((random.random()*2-1) * self.generator.scale_limit_ratio + 1) * scale_ratio
            # rotate and scale
            M = cv2.getRotationMatrix2D(rotate_center, angle, ratio)
            image = cv2.warpAffine(image, M, (fl.get_cols(), fl.get_rows()))
            landmarks = np.transpose(np.dot(M, np.transpose(fl.landmarks_homogenous)))

            x1, y1, x2, y2 = fl.get_bounding_box(self.generator.target_bounding_box_part)
            target_center = ((x1+x2)/2.0, (y1+y2)/2.0, 1)
            target_center = np.dot(M, target_center)
            target_center = (int(round(target_center[0])),
                             int(round(target_center[1])))
            half_target_dim_x = int(self.target_size[0] / 2)
            half_target_dim_y = int(self.target_size[1] / 2)
            # AZ: fix translation to out of image
            # translate
            translate_x = int(round((random.random()*2-1) * self.generator.translate_x_ratio * self.target_size[0]))
            translate_y = int(round((random.random()*2-1) * self.generator.translate_y_ratio * self.target_size[1]))
            translate_x = max(min(image.shape[1]-target_center[0]-half_target_dim_x-1, translate_x),
                              half_target_dim_x - target_center[0])
            translate_y = max(min(image.shape[0]-target_center[1]-half_target_dim_y-1, translate_y),
                              half_target_dim_y - target_center[1])
            image = image[target_center[1] - half_target_dim_y + translate_y: target_center[1] + half_target_dim_y + translate_y,
                          target_center[0] - half_target_dim_x + translate_x: target_center[0] + half_target_dim_x + translate_x]


            landmarks = (landmarks - np.array([target_center[0] - half_target_dim_x + translate_x,
                                               target_center[1] - half_target_dim_y + translate_y]))
            if self.generator.preprocessing_function:
                image = self.generator.preprocessing_function(image)

            values = []
            for i in self.face_landmark_indices:
                values.append(landmarks[i][0])
                values.append(landmarks[i][1])
            X.append(image)
            y.append(values)
        return np.asarray(X), np.asarray(y)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

class ImageFaceLandmarkDataGenerator(object):
    def __init__(self,
                 rotate_bounding_box_part='face',
                 rotate_limit_in_degrees=0,
                 scale_bounding_box_part='face',
                 scale_bounding_box_size=64,
                 scale_limit_ratio=0,
                 translate_x_ratio=0,
                 translate_y_ratio=0,
                 target_bounding_box_part='face',
                 shuffle=True,
                 random_horizontal_flip=False,
                 preprocessing_function=None
                 ):
        self.rotate_bounding_box_part = rotate_bounding_box_part
        self.rotate_limit_in_degrees = rotate_limit_in_degrees
        self.scale_bounding_box_part = scale_bounding_box_part
        self.scale_bounding_box_size = scale_bounding_box_size
        self.scale_limit_ratio = scale_limit_ratio
        self.translate_x_ratio = translate_x_ratio
        self.translate_y_ratio = translate_y_ratio
        self.target_bounding_box_part = target_bounding_box_part
        self.shuffle = shuffle
        self.random_horizontal_flip = random_horizontal_flip
        self.preprocessing_function = preprocessing_function

    def flow(self,
             face_landmark_list,
             face_landmark_indices,
             target_size,
             batch_size,
             seed=None,
             ):
        valid_face_landmark_list = []
        for fl in face_landmark_list:
            landmark_missing = False
            for i in face_landmark_indices:
                if fl.landmarks[i][0] <= 0 or fl.landmarks[i][1] <= 0:
                    landmark_missing = True
                    break
            if not landmark_missing:
                valid_face_landmark_list.append(fl)
        n = len(valid_face_landmark_list)
        assert n > 0, 'no landmarks were found that fits indices: {}'.format(face_landmark_indices)
        return LandmarkDictionaryIterator(valid_face_landmark_list,
                                          face_landmark_indices,
                                          self,
                                          n,
                                          self.shuffle,
                                          self.random_horizontal_flip,
                                          target_size,
                                          batch_size,
                                          seed)
