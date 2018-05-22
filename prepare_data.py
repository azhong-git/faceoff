import os
import cv2
import numpy as np

from face_data import FaceData, FaceLandmark
from face_landmarks import flip_landmarks

def prepare_data(data_path, annotation_path, landmark_type, to_gray = True):
    face_landmark_list = []
    if landmark_type == 'muct_clmtools':
        landmark_size = 71
        with open(annotation_path) as fi:
            for line in fi:
                splitted = line.split(';')
                filename = splitted[0]
                filepath = os.path.join(data_path, filename)
                if not os.path.isfile(filepath):
                    continue
                landmarks = np.array([[float(splitted[i*3+1]), float(splitted[i*3+2])] for i in range(landmark_size)])
                image = cv2.imread(filepath)
                if to_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_landmark = FaceLandmark(image, landmarks, landmark_type)
                face_landmark_list.append(face_landmark)
    elif landmark_type == 'muct':
        landmark_size = 76
        with open(annotation_path) as fi:
            header = True
            for line in fi:
                if header:
                    header = False
                    continue
                splitted = line.split(',')
                filename = splitted[0]
                filepath = os.path.join(data_path, filename + '.jpg')
                if not os.path.isfile(filepath):
                    print('{} not found'.format(filepath))
                    continue
                landmarks = []
                for i in range(landmark_size):
                    x = float(splitted[i*2+2])
                    y = float(splitted[i*2+3])
                    if x == 0 and y == 0:
                        x = -1
                        y = -1
                    landmarks.append([x,y])
                landmarks = np.array(landmarks)
                image = cv2.imread(filepath)
                if to_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_landmark = FaceLandmark(image, landmarks, landmark_type)
                face_landmark_list.append(face_landmark)
    elif landmark_type == 'lfpw':
        landmark_size = 35
        with open(annotation_path) as fi:
            header = True
            for line in fi:
                if header:
                    header = False
                    continue
                splitted = line.split('\t')
                filename = splitted[0].split('/')[-1]
                filepath = os.path.join(data_path, filename)
                # only read average rows
                if (splitted[1] != 'average') or not os.path.isfile(filepath):
                    continue
                landmarks = []
                for i in range(landmark_size):
                    x = float(splitted[i*3+2])
                    y = float(splitted[i*3+3])
                    landmarks.append([x, y])
                landmarks = np.array(landmarks)
                image = cv2.imread(filepath)
                if to_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_landmark = FaceLandmark(image, landmarks, landmark_type)
                face_landmark_list.append(face_landmark)
    else:
        assert False, '{} not supported'
    return face_landmark_list

def augment_data(face_landmarks,
                 augment_size = 10,
                 fliplr = False,
                 rotate_bounding_box_part = 'face',    # rotate around part bounding box
                 rotate_limit_in_degrees = 10,         # rotate limit defaults to -10 to 10 degrees
                 scale_bounding_box_part = 'face',     # scale to fit
                 scale_bounding_box_size = 64,         # scale the larger dimension to fit this
                 scale_limit_ratio = 0.1,              # scale defaults from 0.9 to 1.1
                 translate_bounding_box_part = 'face', # translate base
                 translate_x_ratio = 0.1,              # translate ratio of part bounding box in x direction
                 translate_y_ratio = 0.1,              # translate ratio of part bounding box in y direction
                 output_bounding_box_part = 'face',    # center of output
                 output_size = (64, 64),
                 only_fliplr = False,
                 shrink_x_y = None
):
    import random
    random.seed()
    output_face_landmarks = []
    assert output_size[0] % 2 == 0 and output_size[1] % 2 ==0, 'output size needs to be even'
    for fl in face_landmarks:
        x1, y1, x2, y2 = fl.get_bounding_box(rotate_bounding_box_part)
        rotate_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        x1, y1, x2, y2 = fl.get_bounding_box(scale_bounding_box_part)
        scale_ratio = min(scale_bounding_box_size/1.0/(y2-y1),
                          scale_bounding_box_size/1.0/(x2-x1))
        x1, y1, x2, y2 = fl.get_bounding_box(translate_bounding_box_part)
        translate_x = x2 - x1
        translate_y = y2 - y1
        x1, y1, x2, y2 = fl.get_bounding_box(output_bounding_box_part)
        output_center = (int((x1+x2)/2.0), int((y1+y2)/2.0))
        half_output_dim_x = int(output_size[0] / 2)
        half_output_dim_y = int(output_size[1] / 2)
        for augment_i in range(augment_size):
            # randomly rotate image by -10 to 10 degrees
            angle = (random.random()*2 - 1) * rotate_limit_in_degrees
            # randomly scale image 0.9 - 1.1
            ratio = ((random.random()*2-1) * scale_limit_ratio + 1) * scale_ratio
            M = cv2.getRotationMatrix2D(rotate_center, angle, ratio)
            image = cv2.warpAffine(fl.image, M, (fl.cols, fl.rows))
            landmarks = np.transpose(np.dot(M, np.transpose(fl.landmarks_homogenous)))
            # randomly translate image by ratio
            translate_x = int((random.random()*2-1) * translate_x_ratio * output_size[0] + 0.5)
            translate_y = int((random.random()*2-1) * translate_y_ratio * output_size[1] + 0.5)
            translate_x = max(min(image.shape[1]-output_center[0]-half_output_dim_x-1, translate_x),
                              half_output_dim_x - output_center[0])
            translate_y = max(min(image.shape[0]-output_center[1]-half_output_dim_y-1, translate_y),
                              half_output_dim_y - output_center[1])
            image = image[output_center[1] - half_output_dim_y + translate_y: output_center[1] + half_output_dim_y + translate_y,
                          output_center[0] - half_output_dim_x + translate_x: output_center[0] + half_output_dim_x + translate_x]
            landmarks = (landmarks - np.array([output_center[0] - half_output_dim_x + translate_x,
                                               output_center[1] - half_output_dim_y + translate_y]))
            if not only_fliplr:
                output_face_landmarks.append(FaceLandmark(image, landmarks, fl.landmark_type))
            if fliplr:
                image_flipped = np.fliplr(image)
                landmarks_flipped = (np.array([image.shape[1]-1, 0]) - landmarks) * np.array([1, -1])
                flip_landmarks(landmarks_flipped, fl.landmark_type)
                output_face_landmarks.append(FaceLandmark(image_flipped, landmarks_flipped, fl.landmark_type))

    return np.array(output_face_landmarks)


face_landmarks_dict = {}
# load muct clm data through https://github.com/azhongwl/clmtools; around 600 faces, ~100 in the wild of varying poses
face_landmarks_dict['muct_clmtools'] = prepare_data('/Users/azhong/face/data/muct_clmtools/images/',
                                                    '/Users/azhong/face/data/muct_clmtools/annotations.csv',
                                                    'muct_clmtools')
# load original muct data from https://github.com/StephenMilborrow/muct: 3500 center-looking faces captured in lab with varying lighting conditions
face_landmarks_dict['muct'] = prepare_data('/Users/azhong/face/data/muct/images/',
                                           '/Users/azhong/face/data/muct/muct76-opencv.csv',
                                           'muct')
# load lpfw data https://neerajkumar.org/databases/lfpw: 600 faces in the wild with varying poses
face_landmarks_dict['lfpw_train'] = prepare_data('/Users/azhong/face/data/lfpw_pruned/images/',
                                                 '/Users/azhong/face/data/lfpw_pruned/kbvt_lfpw_v1_train.csv',
                                                 'lfpw')
face_landmarks_dict['lfpw_test'] = prepare_data('/Users/azhong/face/data/lfpw_pruned/images/',
                                                '/Users/azhong/face/data/lfpw_pruned/kbvt_lfpw_v1_test.csv',
                                                'lfpw')

face_landmarks = []
for key in face_landmarks_dict.keys():
    face_landmarks.extend(face_landmarks_dict[key])

for fl in face_landmarks:
    fl.convert_to_landmark_type('muct')
np.random.shuffle(face_landmarks)
augment_size = 5
flip = True
# face_landmarks_augmented = augment_data(face_landmarks,
#                                         augment_size,
#                                         flip,
#                                         'face', 5,
#                                         'face', 64, 0.1,
#                                         'face', 0.2, 0.2,
#                                         'face', (64, 64))
mouth_landmarks_augmented = augment_data(face_landmarks,
                                         augment_size,
                                         flip,
                                         'mouth', 5,
                                         'face', 128, 0.1,
                                         'mouth', 0.2, 0.3,
                                         'mouth', (64, 64),
                                         False)
# left_eye_landmarks_augmented = augment_data(face_landmarks,
#                                             augment_size,
#                                             False,
#                                             'left_eye', 10,
#                                             'face', 128, 0.2,
#                                             'left_eye', 0.3, 0.3,
#                                             'left_eye', (32, 32))
# right_eye_landmarks_augmented = augment_data(face_landmarks,
#                                              augment_size,
#                                              True,
#                                              'right_eye', 10,
#                                              'face', 128, 0.2,
#                                              'right_eye', 0.3, 0.3,
#                                              'right_eye', (32, 32),
#                                              True)
# eye_landmarks_augmented = np.concatenate((left_eye_landmarks_augmented, right_eye_landmarks_augmented))
validation_split = 0.2
if flip == True:
    train_size = int(len(face_landmarks) * augment_size * 2 * (1-validation_split))
else:
    train_size = int(len(face_landmarks) * augment_size * (1-validation_split))
# eye_train_size = int(len(face_landmarks) * augment_size * 2 * (1-validation_split))
# np.random.shuffle(eye_landmarks_augmented[:train_size])
np.random.shuffle(mouth_landmarks_augmented[:train_size])
# np.random.shuffle(face_landmarks_augmented[:train_size])

import pickle
pickle.dump({'train_size': train_size, 'data': mouth_landmarks_augmented},
            open('data/mouth.p', 'wb'))
# pickle.dump({'train_size': eye_train_size, 'data': eye_landmarks_augmented},
#             open('data/eye.p', 'wb'))
# pickle.dump({'train_size': train_size, 'data': face_landmarks_augmented},
#              open('data/face.p', 'wb'))
