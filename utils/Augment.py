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

datagen = ImageDataGenerator()
datagen.fit(trainX)
for batchX, batchy in datagen.flow(trainX, trainy, batch_size=9, seed=99):
# plot first few images
  for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(batchX[i])
  # show the figure
  pyplot.show()
  break
  
choices = int(p*trainX.shape[0])
indices = np.sort(np.random.choice(trainX.shape[0], choices, replace=False))
print(indices)
datagen = ImageDataGenerator(        
            rotation_range=90,
            width_shift_range=0.2,  
            height_shift_range=0.2,    
            shear_range=0.2,        
            zoom_range=0.2,        
            horizontal_flip=True,         
            fill_mode='nearest', cval=125)
tempX = np.zeros((choices,32,32,3))
tempy = np.zeros((choices,1))
c=0
for x in range(trainX.shape[0]):
  if x in indices:
    tempX[c] = trainX[x:x+1]
    tempy[c] = trainy[x:x+1]
    c=c+1

c=0
for batchX,batchy in datagen.flow(tempX,tempy, batch_size=9):    
  for i in range(9):
    pyplot.subplot(330 + 1 + i)
    if i in indices:
      pyplot.imshow(batchX[c])
      c=c+1
    else:
      pyplot.imshow(trainX[i])
  pyplot.show()
  break
  
