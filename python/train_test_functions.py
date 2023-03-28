# =====================================================================================
#
# Basic script for training a tensorflow 2 (keras) image classification model.
# Includes retraining from imagenet weights
# Supports multiple network architectures (InceptionV3, MobileNet, Xception)
#
# Authors:
#   Brent Martin (Manaaki Whenua Landcare Research)
#   Sheldon Coup (University of Canterbury)
#   
#   Adapted by Aaron Harmer (MWLR)
# =====================================================================================

from enum import Enum
import os
from glob import glob
import shutil
import numpy as np
from scipy.special import softmax
import tensorflow as tf # TODO try using tf2 to confirm compatibility?
# from tensorflow import keras

# Keras models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception, preprocess_input

# Keras utilities
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope # Needed because MobileNet uses this custom 'relu6' function

# TODO - needed?
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

# Supported model specifics
# Note: freeze layer def for all layers = (0, num_layers-1)
class Freeze(Enum):
  inception_v3 = (0,312)
  mobilenet = (0,83)
  xception = (0,133)

class Input_Size(Enum):
  inception_v3 = (299,299)
  mobilenet = (224,224)
  xception = (299,299)


# TODO REFACTOR (train and retrain should be/share a single function)
def train(model_name, train_data_dir, valid_data_dir, model_dir, dataset_name, 
            epochs = 250, batch_size = 16, lr = 1e-4, patience = 0):
  """
  learns from scratch
  """

  print("Training a network FROM SCRATCH for dataset '{}'".format(dataset_name))
  print("  Training data:", train_data_dir)
  print("  Validation data:", valid_data_dir)
  print("  Model save dir:", model_dir)
  print("  Model:", model_name)
  print("  Epochs:", epochs)
  print("  Batch size:", batch_size)
  print("  Learning rate:", lr)

  num_classes = len(os.listdir(train_data_dir))

  # Not sure about lopping the top off - not needed???
  if model_name == 'InceptionV3':
    image_size = Input_Size.inception_v3.value
    image_shape = image_size + (3,)
    model = InceptionV3(include_top = True, input_shape = image_shape, classes = num_classes, classifier_activation = "softmax")
  elif model_name == 'Xception':
    image_size = Input_Size.xception.value
    image_shape = image_size + (3,)
    model = Xception(include_top = True, weights = None, input_shape = image_shape, classes = num_classes)
  elif model_name == 'MobileNet':
    image_size = Input_Size.mobilenet.value
    image_shape = image_size + (3,)
    model = MobileNet(include_top = True, input_shape = image_shape, classes = num_classes, classifier_activation = "softmax")
  else:
    print("ERROR: Unknown model {}: supported models are 'InceptionV3', 'MobileNet' and 'Xception'".format(model_name))
    return None
    
  # compile model for training 
  model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  # define data generators
  # Set the augmentations. Small datasets need to be aggressive...
  # TODO: need to experiment with these: could be passed in by caller...
  
  train_datagen = ImageDataGenerator(# rescale = 1./255,
                                     preprocessing_function = preprocess_input, # scales to -1...1 for Xception
                                     rotation_range = 40, # not for larger image set?
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2,
                                     # brightness_range = (0.5,1.5),
                                     channel_shift_range = 20, # [0..255]
                                     horizontal_flip = True)
 
  # valid_datagen = ImageDataGenerator(rescale = 1./255)
  valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

  train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = image_size, batch_size = batch_size)
  valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size = image_size, batch_size = batch_size)
    
  # refit model and return
  print("Retraining {} model...".format(model_name))

  if patience > 0:
    # define early stopping procedure - stop after (patience) epoch if no improvement in the validation set
    # e.g. set patience to 30
    early_stopping_monitor = EarlyStopping(monitor = 'val_acc', verbose = 1, patience = patience)
    callbacks = [early_stopping_monitor]
  else:
    callbacks = None

  model.fit(train_generator, epochs = epochs, validation_data = valid_generator, callbacks = callbacks)

  # Save the model to disk
  model_file_name = os.path.join(model_dir, '{}_{}_{}_epoch_SCRATCH.h5'.format(dataset_name, model_name, epochs))
  model.save(model_file_name)

  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()
  
  print("Done. Model saved to {}".format(model_file_name))
  return model_file_name
  

def retrain(model_name, train_data_dir, valid_data_dir, model_dir, log_dir, dataset_name, 
            epochs = 200, batch_size = 16, lr = 1e-4, fine_tune = True, patience = 30):
  """
  learns a new domain from an imagenet network
  """

  print("Retraining an Imagenet network for dataset '{}'".format(dataset_name))
  print("  Training data:", train_data_dir)
  print("  Validation data:", valid_data_dir)
  print("  Model save dir:", model_dir)
  print("  Model log dir:", log_dir)
  print("  Model:", model_name)
  print("  Epochs:", epochs)
  print("  Batch size:", batch_size)
  print("  Learning rate:", lr)
  print("  Fine_tune:", fine_tune)

  num_classes = len(os.listdir(train_data_dir))
  

  if model_name == 'InceptionV3':
    freeze_between = Freeze.inception_v3.value
    image_size = Input_Size.inception_v3.value
    image_shape = image_size + (3,)
    base_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'Xception':
    freeze_between = Freeze.xception.value
    image_size = Input_Size.xception.value
    image_shape = image_size + (3,)
    base_model = Xception(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'MobileNet':
    freeze_between = Freeze.mobilenet.value
    image_size = Input_Size.mobilenet.value
    image_shape = image_size + (3,)
    base_model = MobileNet(include_top = False, weights = 'imagenet', input_shape = image_shape)
  else:
    print("ERROR: Unknown model {}: supported models are 'InceptionV3', 'MobileNet' and 'Xception'".format(model_name))
    return None

  # Create and randomly initialize the dense top layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  predictions = Dense(num_classes, activation = 'softmax')(x)

  model = Model(inputs = base_model.inputs, outputs = predictions)

  # Retraining: freeze all layers except the new head
  for i in range(freeze_between[0], freeze_between[1]):
    model.layers[i].trainable = False 
    
  # compile model for training 
  model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  # define data generators
  # Set the augmentations. Small datasets need to be aggressive...
  # TODO: should be passed in by caller...

  # Minor augmentation (for wasps)
  train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, # scales to -1...1 for Xception
                                     rotation_range = 25, # Some are tilted (prev 25)
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     zoom_range = 0.1,
                                     # brightness_range = (0.5,1.5)
                                     # channel_shift_range = 20, # [0..255] # not for monochrome
                                     )

                                     
  #valid_datagen = ImageDataGenerator(rescale = 1./255)
  valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

  train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = image_size, batch_size = batch_size)
  valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size = image_size, batch_size = batch_size)
    
  # refit model and return
  print("Retraining {} model...".format(model_name))

  early_stopping_monitor = EarlyStopping(monitor = 'val_accuracy', verbose = 1, patience = patience)
  log_file_name = os.path.join(log_dir, '{}_{}_{}_epoch_transfer_earlyStop_log.csv'.format(dataset_name, model_name, epochs))
  csv_logger = CSVLogger(log_file_name)
  callbacks = [early_stopping_monitor, csv_logger]
  model.fit(train_generator, epochs = epochs, validation_data = valid_generator, callbacks = callbacks)

  # Fine-tune the lower layers (if required)
  if fine_tune:
    # Set the number of fine-tune epochs and learning rate. TODO - is there a smart way to tune this?
    fine_tune_epochs = epochs
    fine_tune_lr = lr/10

    print("Fine-tuning the {} network for a further {} epochs with lr = {}".format(model_name, fine_tune_epochs, fine_tune_lr))
    for i in range(freeze_between[0], freeze_between[1]):
      model.layers[i].trainable = True   

    model.compile(optimizer = tf.keras.optimizers.Adam(fine_tune_lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])     
    model.fit(train_generator, epochs = fine_tune_epochs, validation_data = valid_generator, callbacks = callbacks)
  else:
    print("Fine-tuning NOT REQUESTED")

  # Save the model to disk
  model_file_name = os.path.join(model_dir, '{}_{}_transfer.h5'.format(dataset_name, model_name))
  model.save(model_file_name)

  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()
  
  print("Done. Model saved to {}".format(model_file_name))
  return model_file_name

def retrain_final(model_name, train_data_dir, model_dir, log_dir, dataset_name, 
            epochs = 200, batch_size = 16, lr = 1e-4, fine_tune = True, patience = 30):
  """
  learns a new domain from an imagenet network
  """

  print("Retraining an Imagenet network for dataset '{}'".format(dataset_name))
  print("  Training data:", train_data_dir)
  print("  Model save dir:", model_dir)
  print("  Model log dir:", log_dir)
  print("  Model:", model_name)
  print("  Epochs:", epochs)
  print("  Batch size:", batch_size)
  print("  Learning rate:", lr)
  print("  Fine_tune:", fine_tune)

  num_classes = len(os.listdir(train_data_dir))
  

  if model_name == 'InceptionV3':
    freeze_between = Freeze.inception_v3.value
    image_size = Input_Size.inception_v3.value
    image_shape = image_size + (3,)
    base_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'Xception':
    freeze_between = Freeze.xception.value
    image_size = Input_Size.xception.value
    image_shape = image_size + (3,)
    base_model = Xception(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'MobileNet':
    freeze_between = Freeze.mobilenet.value
    image_size = Input_Size.mobilenet.value
    image_shape = image_size + (3,)
    base_model = MobileNet(include_top = False, weights = 'imagenet', input_shape = image_shape)
  else:
    print("ERROR: Unknown model {}: supported models are 'InceptionV3', 'MobileNet' and 'Xception'".format(model_name))
    return None

  # Create and randomly initialize the dense top layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  predictions = Dense(num_classes, activation = 'softmax')(x)

  model = Model(inputs = base_model.inputs, outputs = predictions)

  # Retraining: freeze all layers except the new head
  for i in range(freeze_between[0], freeze_between[1]):
    model.layers[i].trainable = False 
    
  # compile model for training 
  model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  # define data generators
  # Set the augmentations. Small datasets need to be aggressive...
  # TODO: should be passed in by caller...

  # Minor augmentation (for wasps)
  train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, # scales to -1...1 for Xception
                                     rotation_range = 25, # Some are tilted (prev 25)
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     zoom_range = 0.1,
                                     # brightness_range = (0.5,1.5)
                                     # channel_shift_range = 20, # [0..255] # not for monochrome
                                     )

                                     
  #valid_datagen = ImageDataGenerator(rescale = 1./255)
  # valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

  train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = image_size, batch_size = batch_size)
  # valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size = image_size, batch_size = batch_size)
    
  # refit model and return
  print("Retraining {} model...".format(model_name))

  early_stopping_monitor = EarlyStopping(monitor = 'accuracy', verbose = 1, patience = patience)
  log_file_name = os.path.join(log_dir, '{}_{}_final_transfer_earlyStop_log.csv'.format(dataset_name, model_name))
  csv_logger = CSVLogger(log_file_name)
  callbacks = [early_stopping_monitor, csv_logger]
  model.fit(train_generator, epochs = epochs, callbacks = callbacks)

  # Fine-tune the lower layers (if required)
  if fine_tune:
    # Set the number of fine-tune epochs and learning rate. TODO - is there a smart way to tune this?
    fine_tune_epochs = epochs
    fine_tune_lr = lr/10

    print("Fine-tuning the {} network for a further {} epochs with lr = {}".format(model_name, fine_tune_epochs, fine_tune_lr))
    for i in range(freeze_between[0], freeze_between[1]):
      model.layers[i].trainable = True   

    model.compile(optimizer = tf.keras.optimizers.Adam(fine_tune_lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])     
    model.fit(train_generator, epochs = fine_tune_epochs, callbacks = callbacks)
  else:
    print("Fine-tuning NOT REQUESTED")

  # Save the model to disk
  model_file_name = os.path.join(model_dir, '{}_{}_transfer.h5'.format(dataset_name, model_name))
  model.save(model_file_name)

  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()
  
  print("Done. Model saved to {}".format(model_file_name))
  return model_file_name


def run_model(model_path, labels_path, test_images_path, model_name, ignore = None, model = None, reset = False):
  """
  Runs the model on a folder of images. 
  NOTE: Assumes the images are arranged into folders by class.
  
  - ignore: class number to ignore (typically "Empty")
  """

  # TODO this will be slow if the test image set is too large...
  # Assumes the test images are also arranged into folders by class  
  image_files = sorted(list(glob('{}/*/*.*'.format(test_images_path)))) 
  
  print("\nTesting model {} ({}) on {} ({} images):".format(model_path, model_name, test_images_path, len(image_files)))

  if model_name == 'InceptionV3':
    image_size = Input_Size.inception_v3.value
  elif model_name == 'Xception':
    image_size = Input_Size.xception.value
  elif model_name == 'MobileNet':
    image_size = Input_Size.mobilenet.value
  else:
    print("ERROR: Unknown model '{}': supported models are 'InceptionV3', 'MobileNet' and 'Xception'".format(model_name))
    return None
    
  with open(labels_path, 'r') as f:
    labels = [l for l in f.read().split('\n')]

  # Mobilenet uses a custom relu function so we load it in case it's needed
  if model:
     print("Using existing model")
  else:
    print('Loading model "{}"'.format(model_path))
    # with CustomObjectScope({'relu6': tf.keras.applications.mobilenet.relu6,'DepthwiseConv2D': tf.keras.applications.mobilenet.DepthwiseConv2D}):
    model = tf.keras.models.load_model(model_path)
    print('Model loaded.')
  
  datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

  print('Predicting image classes...')
  data_generator = datagen.flow_from_directory(test_images_path, target_size = image_size, shuffle = False, batch_size = 1)
  predictions = model.predict(data_generator, steps = len(image_files)) # TODO - use glob to get the number of images
  print('Prediction complete.')
  # All done, clear keras' global state to avoid memory leaks
  if reset:
    K.clear_session()

  return labels, predictions, image_files, model


def test_model(model_path, labels_path, test_images_path, model_name, ignore = None):
  """
  Tests the model on a folder of images. 
  NOTE: Assumes the images are arranged into folders by class.
  
  - ignore: class number to ignore (typically "Empty")
  """

  labels, predictions, image_files, model = run_model(model_path, labels_path, test_images_path, model_name, ignore)
  
  # TODO - replace with confusion etc 

  correct = 0
  wrong = 0

  answers = []
  
  print('\nResults:')
  print('---------------------------------------------------------------------------------------------------------')
  print("Classes: {}".format(labels))
  for file_num, p in enumerate(predictions):
    best = 0
    second = 0
    best_index = -1
    for i, pr in enumerate(p):
      # p = softmax(p) # Not used!
      if (not ignore) or (i !=  ignore): # ignore one class (e.g. "empty")
        if pr > best:
          second = best
          best = pr
          best_index = i
        elif pr > second:
          second = pr
    file_path = os.path.relpath(image_files[file_num], test_images_path)
    filename = os.path.basename(file_path)
    class_name = os.path.dirname(file_path)
    answers.append((class_name, labels[best_index], best, file_path, p)) # (actual, predicted)
    if labels[best_index] == class_name:
      outcome = 'CORRECT'
      correct += 1
    else:
      outcome = '***WRONG***'
      wrong += 1
    print('Prediction for {} :{} => {}   {} ({})'.format(file_path, p, labels[best_index], outcome, best/second))

  print("\nResults for model {} ({}) on {} ({} images):".format(model_path, model_name, test_images_path, len(image_files)))
  print('\nCORRECT: {0} ({1:.2f})%  WRONG: {2} ({3:.2f})%'.format(correct, 100* correct/(correct + wrong), wrong, 100* wrong/(correct + wrong))) 
  print('---------------------------------------------------------------------------------------------------------')
  
  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()

  return labels, answers


def predict_new(model_path, labels_path, test_images_path, model_name, ignore = None):
  """
  Predict the ID of a folder of images. 
  
  """

  labels, predictions, image_files, model = run_model(model_path, labels_path, test_images_path, model_name, ignore)

  answers = []
  
  print('\nResults:')
  print('---------------------------------------------------------------------------------------------------------')
  print("Classes: {}".format(labels))
  for file_num, p in enumerate(predictions):
    best = 0
    second = 0
    best_index = -1
    for i, pr in enumerate(p):
      if (not ignore) or (i !=  ignore): # ignore one class (e.g. "empty")
        if pr > best:
          second = best
          best = pr
          best_index = i
        elif pr > second:
          second = pr
    file_path = os.path.relpath(image_files[file_num], test_images_path)
    filename = os.path.basename(file_path)
    class_name = os.path.dirname(file_path)
    answers.append((class_name, labels[best_index], best, best/second, image_files[file_num], p)) # (actual, predicted)
    print('Prediction for {}: => {} {} ({})'.format(file_path, labels[best_index], best, best/second))
  print('---------------------------------------------------------------------------------------------------------')
  
  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()

  return labels, answers
