from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

path = r"training/messi/10.jpg"

img = image.load_img(path)
plt.imshow(img)
cv2.imread(path).shape

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('training/',
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='binary')