data_path = '/disk/multi-pie/Multi-Pie/data/'
annotation_path = '/data/multipie/labels'

import cv2
import scipy.io as sio
import glob
import os
import re
import numpy as np

camera_labels = sorted(os.listdir(annotation_path))
camera_labels = [s for s in camera_labels if not s.startswith('.')]
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
        image_filename = os.path.join(data_path,
                                      'session'+session_number,
                                      'multiview',
                                      subject_id,
                                      recording_number,
                                      camera_label[0:2]+'_'+camera_label[2],
                                      part_name+'.png')
        image_filenames = glob.glob(image_filename[:-6] + '*')
        if camera_label == '081' or camera_label == '191':
            for filename in image_filenames:
                image = cv2.imread(filename)
                image = np.flipud(image)
                cv2.imwrite(filename, image)
