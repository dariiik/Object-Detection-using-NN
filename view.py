# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from glob import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from google.colab import drive
drive.mount('/content/gdrive/')

#Train Method

train_dataset_path = 'gdrive/MyDrive/AIP2_FinalProject/train.npy' 

valid_ratio = 0.2 
train_np_data = np.load(train_dataset_path, allow_pickle=True) 
data, target = np.asarray(train_np_data.item().get('data')), np.asarray(train_np_data.item().get('target')) 

#Explore the dataset
print(f"total number of img data : {data.shape[0]}")
print(f"total number of label data : {target.shape[0]}")

sample_data, sample_target = data[0], target[0]
print(sample_data.shape) 
print(sample_target.shape) 

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size = valid_ratio)

%matplotlib inline

action_dict = {'0': 'sitting', '1': 'sleeping', '2': 'running', '3': 'cycling', '4': 'texting', '5': 'calling', '6': 'eating',
         '7': 'clapping', '8': 'drinking', '9': 'hugging', '10': 'using_laptop', '11': 'laughing', '12': 'listening_to_music', '13': 'fighting' , '14': 'dancing'}

img_order = 500 
sample_data = data[img_order] 
sample_data = sample_data.astype(int)
sample_target = target[img_order]
sample_action = action_dict[str(np.argmax(sample_target))] 

print(f"Action : {sample_action}")
plt.imshow(sample_data)

STEP_SIZE_TRAIN = X_train.shape[0]//batch_size
STEP_SIZE_VALID = X_valid.shape[0]//batch_size

print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)
image_height = 64
image_width = 64
image_channels = 3

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, image_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(15, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
import cv2

resized_X_train = np.array([cv2.resize(img, (64, 64)) for img in X_train])
resized_X_valid = np.array([cv2.resize(img, (64, 64)) for img in X_valid])

print(resized_X_train.shape)  
print(resized_X_valid.shape) 
history = model.fit(resized_X_train, y_train, epochs=n_epoch, batch_size=batch_size, validation_data=(resized_X_valid, y_valid))

test_np_data = np.load(test_dataset_path, allow_pickle=True) 
data, name = np.asarray(test_np_data.item().get('data')), np.asarray(test_np_data.item().get('name')) 
resized_data = np.array([cv2.resize(img, (64, 64)) for img in data])


print(f"total number of img data : {data.shape[0]}")
print(f"shape of img : {data.shape[1:]}")
print(resized_data.shape)
output = model.predict(resized_data) 
output = np.argmax(output, axis=-1)  
print(output)
data = {
    'img' : name,
    'predictions' : output
}


def _key(_str): 
  str2int = int(_str.split('_')[1].split('.')[0])
  return str2int

output_df = pd.DataFrame(data) 
output_df['order'] = output_df['img'].apply(_key) 
output_df = output_df.sort_values(by = 'order')
output_df = output_df[['img', 'predictions']].reset_index(drop = True) 

output_df.to_csv('result.csv')
