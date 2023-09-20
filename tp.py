from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

path = r"C:\Users\Mili\Desktop\tp2\training\messi\10.jpg"

img = image.load_img(path)
plt.imshow(img)