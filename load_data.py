import pickle
import numpy as np

def load_landmarks_data(pickle_path):
    face_landmarks_data = pickle.load(open(pickle_path, 'rb'))
    images = []
    values = []
    for fl in face_landmarks_data:
        images.append(fl.image.astype('float32'))
        abs_open = fl.get_distance(57, 60)
        lr_dist = fl.get_distance(44, 50)
        values.append(abs_open / lr_dist)
    images = np.asarray(images)
    images = np.expand_dims(images, -1)
    images = (images/255.0 - 0.5) * 2.0
    return images, values

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
