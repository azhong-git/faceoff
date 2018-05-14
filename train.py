from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

import datetime
import os

from load_data import load_landmarks_data, split_data
from models import simple_CNN, simpler_CNN, simple_nodropout_CNN, simpler_nodropout_CNN, mini_inception, Kao_Onet

batch_size = 32
input_shape = (32, 32, 1)
output_size = 12
num_epochs = 1000
# validation split has to be 0.2, see prepare_data.py
validation_split = .2
patience = 50

now = datetime.datetime.now()
model_name = 'kao_onet_{}_lm_{}'.format(input_shape[0], output_size)
base_path = 'models/' + model_name + '_'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

# load images and vals
images, vals = load_landmarks_data('data/mouth.p', (input_shape[0], input_shape[1]))

# define CNN model
model = Kao_Onet(input_shape, output_size)
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.summary()

# training params
log_file_path = base_path + 'faceoff_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'faceoff_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

train_data, val_data = split_data(images, vals, validation_split)
train_images, train_vals = train_data
model.fit(x = train_images,
          y = train_vals,
          batch_size = batch_size,
          epochs = num_epochs,
          callbacks = callbacks,
          verbose = 1,
          validation_data = val_data
)
