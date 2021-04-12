#Loading the cifar10 dataset from Keras.datasets
from keras.datasets import cifar10

#importing some basic modules
import numpy as np
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# load dataset into train and test sets
(trainX, trainy), (testX, testy) = cifar10.load_data()

# show shape of loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX/=255
testX/=255

def augment_data( trainX, trainy, p=0.25, batch_size=32):
  num_choices = int(p*trainX.shape[0])
  
  #Selecting random image indices to be augmented in each step.
  indices = np.zeros((8,num_choices))
  for i in range(8):
    indices[i] = np.sort(np.random.choice(trainX.shape[0], num_choices, replace=False))
  indices= indices. astype(int)

  total_iterations = int(trainX.shape[0] / batch_size) + 1
  
  #Augmentation Pipeline: composed of 8 steps.
  for i in range(8):
    num_augment = np.zeros(8).astype(int)
    num_augment[i] = 1
    datagen = ImageDataGenerator( rotation_range=150*num_augment[0],
                                  horizontal_flip=bool(num_augment[1]), 
                                  vertical_flip = bool(num_augment[2]),
                                  width_shift_range=0.3* num_augment[3],
                                  height_shift_range=0.3* num_augment[4],
                                  shear_range=0.3* num_augment[5],
                                  zoom_range=0.3* num_augment[6],
                                  samplewise_std_normalization= bool(num_augment[7]),
                                  fill_mode='nearest')
    
    for iter in range(total_iterations):
      #temporarily Storing images to be augmented in this ndarray in 
      # for each iteration
      tempX = np.zeros((batch_size,32,32,3))
      tempy = np.zeros((batch_size,1))
      start_index = batch_size * iter
      end_index = start_index + batch_size

      for x in range(start_index, end_index):
        if x in indices[i]:
          tempX[x%batch_size] = trainX[x:x+1]
          tempy[x%batch_size] = trainy[x:x+1]
      #Applying datagen on temp images to be augmented, in each iteration.
      datagen.fit(tempX)
      for batchX,batchy in datagen.flow(tempX,tempy, batch_size,shuffle=False):    
        for x in range(start_index, end_index):
          if x in indices[i]:
            trainX[x] = batchX[x%batch_size]
        break

  return trainX, trainy
