import os
import cv2
import numpy as np
import re
import pickle
from sys import platform

from face_data import FaceData, FaceLandmark
from face_landmarks import flip_landmarks

MULTIPIE_SELECT_LIGHTING = 5
to_gray = False
use_preload_images = True
if to_gray:
    color = 'gray'
else:
    color = 'bgr'

class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def prepare_data(data_path, annotation_path, landmark_type):
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
                face_landmark = FaceLandmark(image, landmarks, landmark_type)
                face_landmark_list.append(face_landmark)
    elif landmark_type == 'multipie':
        import scipy.io as sio
        import glob
        import random
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
                    image = cv2.imread(image_filename_picked)
                    if camera_label == '081' or camera_label == '191':
                        image = np.flipud(image)
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
    size = len(face_landmarks)
    for fl_ind in range(size, 0, -1):
        fl = face_landmarks[fl_ind-1]
        # fl = face_landmarks.pop(-1)
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


if not use_preload_images or not os.path.isfile('data/face_landmarks_{}.p'.format(color)):
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
    # load multipie data: 5302 faces x 20 lighting conditions from various poses
    face_landmarks_dict['multipie'] = prepare_data('/Volumes/NO NAME/Multi-Pie/data/',
                                                  '/Users/azhong/face/data/multipie/labels',
                                                   'multipie')
    face_landmarks = []
    for key in face_landmarks_dict.keys():
        face_landmarks.extend(face_landmarks_dict[key])
    for fl in face_landmarks:
        fl.convert_to_landmark_type('muct')
    np.random.shuffle(face_landmarks)
    # print('before dumping')
    # if platform == 'darwin':
    #     pickle.dump(face_landmarks, MacOSFile(open('data/face_landmarks_bgr.p', 'wb')), protocol = 4)
    # else:
    #     pickle.dump(face_landmarks, open('data/face_landmarks_bgr.p', 'wb'), protocol = 4)
    # print('converting...')
    if to_gray:
        for fl in face_landmarks:
            fl.cvtColor(cv2.COLOR_BGR2GRAY)
    # if platform == 'darwin':
    #     pickle.dump(face_landmarks, MacOSFile(open('data/face_landmarks_gray.p', 'wb')), protocol = 4)
    # else:
    #     pickle.dump(face_landmarks, open('data/face_landmarks_gray.p', 'wb'), protocol = 4)
else:
    if platform == 'darwin':
        face_landmarks = pickle.load(MacOSFile(open('data/face_landmarks_bgr.p', 'rb')))
    else:
        face_landmarks = pickle.load(open('data/face_landmarks_bgr.p', 'rb'))
    for fl in face_landmarks:
        fl.cvtColor(cv2.COLOR_BGR2GRAY)
    if platform == 'darwin':
        pickle.dump(face_landmarks, MacOSFile(open('data/face_landmarks_gray.p', 'rb')), protocol = 4)
    else:
        pickle.dump(face_landmarks, open('data/face_landmarks_gray.p', 'rb'), protocol = 4)

augment_size = 4
flip = True
face_width = 64
object_width = 32
validation_split = 0.2
if flip == True:
    train_size = int(len(face_landmarks) * augment_size * 2 * (1-validation_split))
else:
    train_size = int(len(face_landmarks) * augment_size * (1-validation_split))
# eye_train_size = int(len(face_landmarks) * augment_size * 2 * (1-validation_split))
print('start augmenting data')
# face_landmarks_augmented = augment_data(face_landmarks,
#                                         augment_size,
#                                         flip,
#                                         'face', 10,
#                                         'face', face_width, 0.1,
#                                         'face', 0.2, 0.2,
#                                         'face', (object_width, object_width))
mouth_landmarks_augmented = augment_data(face_landmarks,
                                         augment_size,
                                         flip,
                                         'mouth', 10,
                                         'face', face_width, 0.1,
                                         'mouth', 0.2, 0.3,
                                         'mouth', (object_width, object_width))
print('face: {}'.format(len(face_landmarks)))
# left_eye_landmarks_augmented = augment_data(face_landmarks,
#                                             augment_size,
#                                             False,
#                                             'left_eye', 10,
#                                             'face', face_width, 0.2,
#                                             'left_eye', 0.3, 0.3,
#                                             'left_eye', (eye_width, eye_width))
# right_eye_landmarks_augmented = augment_data(face_landmarks,
#                                              augment_size,
#                                              True,
#                                              'right_eye', 10,
#                                              'face', face_width, 0.2,
#                                              'right_eye', 0.3, 0.3,
#                                              'right_eye', (eye_width, eye_width),
#                                              True)
# eye_landmarks_augmented = np.concatenate((left_eye_landmarks_augmented, right_eye_landmarks_augmented))
print('finished augmenting data')
np.random.shuffle(mouth_landmarks_augmented[:train_size])
# np.random.shuffle(face_landmarks_augmented[:train_size])
# np.random.shuffle(eye_landmarks_augmented[:eye_train_size])

if platform == 'darwin':
    pickle.dump({'train_size': train_size, 'data': mouth_landmarks_augmented},
                MacOSFile(open('data/mouth_{}_{}_{}.p'.format(color, face_width, object_width), 'wb')), protocol=4)
    # pickle.dump({'train_size': train_size, 'data': face_landmarks_augmented},
    #             MacOSFile(open('data/face.p', 'wb')), protocol=4)
    # pickle.dump({'train_size': eye_train_size, 'data': eye_landmarks_augmented},
    #             MacOSFile(open('data/eye_{}_{}_{}.p'.format(color, face_width, eye_width), 'wb')), protocol=4)
else:
    pickle.dump({'train_size': train_size, 'data': mouth_landmarks_augmented},
                open('data/mouth_{}_{}_{}.p'.format(color, face_width, object_width), 'wb'), protocol=4)
    # pickle.dump({'train_size': train_size, 'data': face_landmarks_augmented},
    #             open('data/face.p', 'wb'), protocol=4)
    # pickle.dump({'train_size': eye_train_size, 'data': eye_landmarks_augmented},
    #             open('data/eye_{}_{}_{}.p'.format(color, face_width, eye_width), 'wb'), protocol=4)
