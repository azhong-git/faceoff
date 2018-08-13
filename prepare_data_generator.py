import os
import cv2
import numpy as np
import re
import pickle
from sys import platform

from face_data import FaceData, FaceLandmark
from face_landmarks import flip_landmarks

MULTIPIE_SELECT_LIGHTING = 4
to_gray = False
use_preload_images = True
if to_gray:
    color = 'gray'
else:
    color = 'bgr'

def prepare_data(data_path, annotation_path, landmark_type, multipie_seed = None, multipie_skip_incomplete = False):
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
                face_landmark = FaceLandmark(landmarks, landmark_type)
                face_landmark.set_image_path(filepath)
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
                face_landmark = FaceLandmark(landmarks, landmark_type)
                face_landmark.set_image_path(filepath)
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
                face_landmark = FaceLandmark(landmarks, landmark_type)
                face_landmark.set_image_path(filepath)
                face_landmark_list.append(face_landmark)
    elif landmark_type == 'multipie':
        import scipy.io as sio
        import glob
        import random
        random.seed(multipie_seed)
        camera_labels = sorted(os.listdir(annotation_path))
        camera_labels = [s for s in camera_labels if not s.startswith('.')]
        image_count = 0
        for camera_label in camera_labels:
            label_filenames = sorted(os.listdir(os.path.join(annotation_path, camera_label)))
            for label_filename in label_filenames:
                re_result = re.search('(.+)_(.+)_(.+)_(.+)_(.+)_lm.mat', label_filename)
                part_name = label_filename[:-7]
                assert re_result
                subject_id = re_result.group(1)
                session_number = re_result.group(2)
                recording_number = re_result.group(3)
                assert camera_label == re_result.group(4)
                image_number = re_result.group(5)
                full_filename = os.path.join(annotation_path, camera_label, label_filename)
                assert os.path.isfile(full_filename)
                landmarks = sio.loadmat(full_filename)['pts']
                if multipie_skip_incomplete and len(landmarks) != 68:
                    continue
                if len(landmarks) != 68 and len(landmarks) != 66:
                    continue
                image_filename = os.path.join(data_path,
                                              'session'+session_number,
                                              'multiview',
                                              subject_id,
                                              recording_number,
                                              camera_label[0:2]+'_'+camera_label[2],
                                              part_name+'.png')
                image_filenames = glob.glob(image_filename[:-6] + '*')
                random.shuffle(image_filenames)
                image_count += 1
                if (image_count % 500) == 499:
                    print('Loading Multipie ...', image_count)
                #     return face_landmark_list # only for debugging
                for image_filename_picked in image_filenames[0:MULTIPIE_SELECT_LIGHTING]:
                    face_landmark = FaceLandmark(landmarks, landmark_type)
                    face_landmark.set_image_path(image_filename_picked)
                    face_landmark_list.append(face_landmark)
    else:
        assert False, '{} not supported'
    return face_landmark_list
