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
# load muct clm data through https://github.com/azhongwl/clmtools; around 600 faces, ~100 in the wild of varying poses
landmarks = prepare_data('{}/muct_clmtools/images/'.format(data_dir),
                         '{}/muct_clmtools/annotations.csv'.format(data_dir),
                         'muct_clmtools')
face_landmarks_dict['train']['muct_clmtools'], face_landmarks_dict['val']['muct_clmtools'] = shuffle_and_split(
    landmarks, validation_split, seed)

# load original muct data from https://github.com/StephenMilborrow/muct: 3500 center-looking faces captured in lab with varying lighting conditions
landmarks = prepare_data('{}/muct/images/'.format(data_dir),
                         '{}/muct/muct76-opencv.csv'.format(data_dir),
                         'muct')
face_landmarks_dict['train']['muct'], face_landmarks_dict['val']['muct'] = shuffle_and_split(
    landmarks, validation_split, seed)

# load lpfw data https://neerajkumar.org/databases/lfpw: 600 faces in the wild with varying poses
landmarks = prepare_data('{}/lfpw_pruned/images/'.format(data_dir),
                         '{}/lfpw_pruned/kbvt_lfpw_v1_train.csv'.format(data_dir),
                         'lfpw')
face_landmarks_dict['train']['lfpw_train'], face_landmarks_dict['val']['lfpw_train'] = shuffle_and_split(
    landmarks, validation_split, seed)
landmarks = prepare_data('{}/lfpw_pruned/images/'.format(data_dir),
                         '{}/lfpw_pruned/kbvt_lfpw_v1_test.csv'.format(data_dir),
                         'lfpw')
face_landmarks_dict['train']['lfpw_test'], face_landmarks_dict['val']['lfpw_test'] = shuffle_and_split(
    landmarks, validation_split, seed)

# load multipie data: 5302 faces x 20 lighting conditions from various poses
landmarks = prepare_data('/disk/multi-pie/Multi-Pie/data/',
                         '{}/multipie/labels'.format(data_dir),
                         'multipie')
face_landmarks_dict['train']['multipie'], face_landmarks_dict['val']['multipie'] = shuffle_and_split(
    landmarks, validation_split, seed)

for train_val in ['train', 'val']:
    for key in face_landmarks_dict[train_val].keys():
        face_landmarks[train_val].extend(face_landmarks_dict[train_val][key])
for train_val in ['train', 'val']:
    for fl in face_landmarks[train_val]:
        fl.convert_to_landmark_type('muct')
train_size = len(face_landmarks['train'])
val_size = len(face_landmarks['val'])
print('train size: ', train_size, ' val size: ', val_size)

### training
import datetime
import os
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from generator import ImageFaceLandmarkDataGenerator
from muct import MUCT

def preprocess_image(image):
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    return image

train_data_gen = ImageFaceLandmarkDataGenerator(rotate_bounding_box_part='face',
                                                rotate_limit_in_degrees=10,
                                                scale_bounding_box_part='face',
                                                scale_bounding_box_size=64,
                                                scale_limit_ratio=0.1,
                                                translate_x_ratio=0.2,
                                                translate_y_ratio=0.3,
                                                target_bounding_box_part='mouth',
                                                preprocessing_function=preprocess_image)
val_data_gen = ImageFaceLandmarkDataGenerator(rotate_bounding_box_part='face',
                                              scale_bounding_box_part='face',
                                              scale_bounding_box_size=64,
                                              target_bounding_box_part='mouth',
                                              preprocessing_function=preprocess_image)
batch_size = 32
input_shape = (48, 48, 3)
topic = 'mouth_12'
output_size = int(topic.split('_')[-1])
num_epochs = 1000
patience = 50

now = datetime.datetime.now()
model_name = 'kao_onet_{}x{}x{}_{}_batch_{}'.format(input_shape[0], input_shape[1], input_shape[2], topic, batch_size)

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
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'faceoff_' + model_name
model_names = trained_models_path + '.{epoch:03d}-{val_loss:.2f}-{loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

model.fit_generator(train_data_gen.flow(face_landmarks['train'], MUCT[topic], (input_shape[0], input_shape[1]), batch_size),
                    steps_per_epoch = int(train_size/batch_size),
                    epochs = num_epochs,
                    callbacks = callbacks,
                    verbose = 1,
                    validation_data = val_data_gen.flow(face_landmarks['val'], MUCT[topic], (32, 32), batch_size))
