import pickle
import numpy as np
import cv2

def load_landmarks_data(pickle_path, input_shape = (64, 64)):
    face_landmarks_data = pickle.load(open(pickle_path, 'rb'))
    images = []
    values = []
    for fl in face_landmarks_data:
        if (fl.rows, fl.cols) != input_shape:
            image = cv2.resize(fl.image, (input_shape[1], input_shape[0]))
        else:
            image = fl.image
        images.append(image.astype('float32'))
        ## distances: not well learnt by network
        # abs_open = fl.get_distance(57, 60)
        # lr_dist = fl.get_distance(44, 50)
        # tb_dist = fl.get_distance(47, 53)
        # values.append(abs_open/lr_dist)
        # values.append(np.array([abs_open, tb_dist]))
        ## direct landmarks
        landmark_indices = [48, 54, 51, 57, 64, 61]
        value = []
        for i in landmark_indices:
            value.append(fl.landmarks[i][0] / 1.0 / fl.cols * input_shape[1])
            value.append(fl.landmarks[i][1] / 1.0 / fl.rows * input_shape[0])
        values.append(value)
    images = np.asarray(images)
    images = np.expand_dims(images, -1)
    images = (images/255.0 - 0.5) * 2.0
    return np.array(images), np.array(values)

def split_data(x, y, split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
