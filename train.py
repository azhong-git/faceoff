from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

import datetime
import os

from load_data import load_landmarks_data, split_data
from models import simple_CNN, simpler_CNN, XCEPTION_CNN

batch_size = 32
input_shape = (64, 64, 1)
num_epochs = 100
# validation split has to be 0.2, see prepare_data.py
validation_split = .2
patience = 50

now = datetime.datetime.now()
model_name = 'simpler_cnn'
base_path = 'models/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

# load images and vals
images, vals = load_landmarks_data('data/mouth.p')

# define CNN model
model = simpler_CNN((64, 64, 1))
model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
model.summary()

# training params
log_file_path = base_path + 'faceoff_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'faceoff_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
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
