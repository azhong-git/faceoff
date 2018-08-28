import os
from prepare_data_generator import prepare_data

### data preparation
import random
seed = 32
validation_split = 0.2

def shuffle_and_split(data, validation_split, seed):
    import random
    random.seed(seed)
    random.shuffle(data)
    random.seed()
    split = int(len(data)*(1-validation_split))
    return data[:split], data[split:]

face_landmarks_dict = {'train': {}, 'val': {}}
face_landmarks     = {'train': [], 'val': []}

data_dir = '/data/'
# load multipie data: 5302 faces x 20 lighting conditions from various poses
# landmarks = prepare_data('/disk/multi-pie/Multi-Pie/data/',
#                          '{}/multipie/labels'.format(data_dir),
#                          'multipie', multipie_seed = seed, multipie_skip_incomplete=True)
# face_landmarks_dict['train']['multipie'], face_landmarks_dict['val']['multipie'] = shuffle_and_split(
#     landmarks, validation_split, seed)
## load 300W-LP data
landmarks = prepare_data('{}/300W_LP'.format(data_dir),
                         '{}/300W_LP/landmarks'.format(data_dir),
                         '300W-LP', LP300W_3D=True)
face_landmarks_dict['train']['300W-LP'], face_landmarks_dict['val']['300W-LP'] = shuffle_and_split(
    landmarks, validation_split, seed)

for train_val in ['train', 'val']:
    for key in face_landmarks_dict[train_val].keys():
        face_landmarks[train_val].extend(face_landmarks_dict[train_val][key])

train_size = len(face_landmarks['train'])
val_size = len(face_landmarks['val'])
print('train size: ', train_size, ' val size: ', val_size)

### training
import datetime
import os
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from generator import ImageFaceLandmarkDataGenerator
from landmark_descriptor import MUCT, MULTIPIE

def preprocess_image(image):
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    return image

topic = os.environ['topic']
part = '_'.join(topic.split('_')[:-1])
scale_limit_ratio=float(os.environ['scale_limit_ratio'])
translate_x_ratio=float(os.environ['translate_x_ratio'])
translate_y_ratio=float(os.environ['translate_y_ratio'])
reduce_lr_factor=float(os.environ['reduce_lr_factor'])

batch_size = int(os.environ['batch_size'])
scale_face = int(os.environ['scale_face'])
input_shape = (os.environ['input_shape']).split('x')
input_shape = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
random_horizontal_flip = False
output_size = int(topic.split('_')[-1])
num_epochs = 1000
patience = 50

train_data_gen = ImageFaceLandmarkDataGenerator(rotate_bounding_box_part='face',
                                                rotate_limit_in_degrees=10,
                                                scale_bounding_box_part='face',
                                                scale_bounding_box_size=scale_face,
                                                scale_limit_ratio=scale_limit_ratio,
                                                translate_x_ratio=translate_x_ratio,
                                                translate_y_ratio=translate_y_ratio,
                                                target_bounding_box_part=part,
                                                random_horizontal_flip=random_horizontal_flip,
                                                preprocessing_function=preprocess_image)
val_data_gen = ImageFaceLandmarkDataGenerator(rotate_bounding_box_part='face',
                                              scale_bounding_box_part='face',
                                              scale_bounding_box_size=scale_face,
                                              target_bounding_box_part=part,
                                              preprocessing_function=preprocess_image)

now = datetime.datetime.now()
model_name = 'kao_onet_{}x{}x{}_face_{}_{}_batch_{}_random_hflip_{}_reduce_lr_{}'.format(input_shape[0], input_shape[1], input_shape[2], scale_face, topic, batch_size, random_horizontal_flip, reduce_lr_factor)

base_path = 'models/' + model_name + '_'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

from models import Kao_Onet
model = Kao_Onet(input_shape, output_size)
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])

log_file_path = base_path + 'faceoff_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=reduce_lr_factor,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'faceoff_' + model_name
model_names = trained_models_path + '.{epoch:03d}-{val_loss:.2f}-{loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

model.fit_generator(train_data_gen.flow(face_landmarks['train'], MULTIPIE[topic], (input_shape[0], input_shape[1]), batch_size),
                    steps_per_epoch = int(train_size/batch_size),
                    epochs = num_epochs,
                    callbacks = callbacks,
                    verbose = 1,
                    validation_data = val_data_gen.flow(face_landmarks['val'], MULTIPIE[topic], (input_shape[0], input_shape[1]), batch_size))
