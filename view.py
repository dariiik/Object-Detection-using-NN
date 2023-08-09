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
#Mount your google drive
from google.colab import drive
drive.mount('/content/gdrive/')

#Train Method

#Upload train dataset
train_dataset_path = 'gdrive/MyDrive/AIP2_FinalProject/train.npy' #Type your train dataset path. Check your own dataset path before typing it. /// ex) 'gdrive/MyDrive/AIP2_FinalProject/train.npy'
#test_dataset_path =  #Type your test dataset path. Check your own dataset path before typing it. /// ex) 'gdrive/MyDrive/AIP2_FinalProject/test.npy'
#You can refer the tutorial ppt if you want to know more details about this

valid_ratio = 0.2 # Type the ratio for splition train data into train and valid. /// ex) the number of data = 100, valid_ratio = 0.2 --> the number of train data = 80, the number of valid data = 20
#batch_size = #Type your own batch size
#n_epoch = #Type your Own Epoch num

train_np_data = np.load(train_dataset_path, allow_pickle=True) #Upload train data. train_np_data consists of img(data) and label(target).
data, target = np.asarray(train_np_data.item().get('data')), np.asarray(train_np_data.item().get('target')) #Get img data(data) and label(target) from train_np_data.

#Explore the dataset
print(f"total number of img data : {data.shape[0]}")
print(f"total number of label data : {target.shape[0]}")

sample_data, sample_target = data[0], target[0]
print(sample_data.shape) # (160, 160, 3) --> img size is 160 x 160 and its type is RGB image(3 channel).
print(sample_target.shape) # (15, ) --> one-hot encoding for label. /// ex)  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] --> the solution label is the 4th label.

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size = valid_ratio) # Split train data into train and valid according to the split ratio.

#Visualize the image sample
%matplotlib inline

action_dict = {'0': 'sitting', '1': 'sleeping', '2': 'running', '3': 'cycling', '4': 'texting', '5': 'calling', '6': 'eating',
         '7': 'clapping', '8': 'drinking', '9': 'hugging', '10': 'using_laptop', '11': 'laughing', '12': 'listening_to_music', '13': 'fighting' , '14': 'dancing'} # Dictionary used for converting one-hot encoding to genre name.

img_order = 500 # You can select any image you want by changing the variable 'img_order'.
sample_data = data[img_order] #Get the image in data.
sample_data = sample_data.astype(int)
sample_target = target[img_order] #Get the target about the selected image. it is the state of one-hot encoding.
sample_action = action_dict[str(np.argmax(sample_target))] #Convert one-hot encoding to genre name

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
# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
import cv2

# Resize input images to match the expected size of 64x64 pixels
resized_X_train = np.array([cv2.resize(img, (64, 64)) for img in X_train])
resized_X_valid = np.array([cv2.resize(img, (64, 64)) for img in X_valid])

# Verify the new shapes
print(resized_X_train.shape)  # (new_number_of_samples, 64, 64, 3)
print(resized_X_valid.shape)  # (new_number_of_samples, 64, 64, 3)
history = model.fit(resized_X_train, y_train, epochs=n_epoch, batch_size=batch_size, validation_data=(resized_X_valid, y_valid))
### Test method ###

test_np_data = np.load(test_dataset_path, allow_pickle=True) #Upload test data. test_np_data consists of img(data) and img name(name).
data, name = np.asarray(test_np_data.item().get('data')), np.asarray(test_np_data.item().get('name')) ##Get img data(data) and name from test_np_data.
resized_data = np.array([cv2.resize(img, (64, 64)) for img in data])


#Explore the dataset
print(f"total number of img data : {data.shape[0]}")
print(f"shape of img : {data.shape[1:]}")
print(resized_data.shape)
output = model.predict(resized_data)  # Predict the result with your trained model.
output = np.argmax(output, axis=-1)  # Convert one-hot encoding to label
print(output)
# Save your result to csv file.
data = {
    'img' : name,
    'predictions' : output
}

# You should submit the saved `result.csv' to the codalab.

def _key(_str): #extract the order from img_name
  # print(_str)
  str2int = int(_str.split('_')[1].split('.')[0])
  return str2int

output_df = pd.DataFrame(data) # Create dataframe
output_df['order'] = output_df['img'].apply(_key) #the 'order' column is used for sorting output in descending order.
output_df = output_df.sort_values(by = 'order')
output_df = output_df[['img', 'predictions']].reset_index(drop = True) #Extract colunms necessary to evaluate in Codalab.

output_df.to_csv('result.csv')
