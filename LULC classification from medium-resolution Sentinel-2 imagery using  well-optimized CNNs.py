#!/usr/bin/env python
# coding: utf-8

# # Land Use and Land Cover Classification from Sentinel-2 imagery using Deep Convolutional Neural Networks

# In this tutorial, we will perform the classification of Sentinel-2 multispectral images using well-optimized Convolutional Neural Networks (CNNs). The process involves training a well-optimized CNN model and subsequently applying the pre-trained model to map Land Use and Land Cover (LULC) in Sentinel-2 imagery. All the implementation will be carried out using Python programming within the Google Colab Pro environment, and the CNNs will be implemented using the Keras open-source Deep Learning library.

# # Dataset Description

# Five different LULC classes—settlement, barren land, fallow land, vegetation, and water bodies—are considered for classification of the Sentinel-2 composite images 4-band. For image classification, we employ a patch-based CNN approach, known for its superior performance compared to the pixel-based CNN method in terms of classification accuracy. The CNN patch size is determined based on the LULC features to be extracted from the satellite imagery. This includes considerations for their size, spatial structure, and the image's spatial resolution. A 5 * 5 pixel patch size was chosen due to the complex structure of LULC features. Moving forward, training data was collected. In total, 2400 training patches, and 600 testing patches of 5 * 5 pixel dimensions were extracted for each LULC class. These training patches were manually labeled through visual interpretation using high-resolution Google Earth imagery. 

# In[ ]:


#import Google drive in Google Colab 
from google.colab import drive
drive.mount('/content/gdrive')


# In[14]:


#Setting up environment in Google Colab 

# import all necessary libraries for CNN model architecture and training 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import optimizers


# Now we will import the training and testing dataset from Google drive. Training and testing data is in numpy files. So, we will first import the numpy library.

# In[15]:


#import training and testing dataset
import numpy as np
X_train=np.load('/content/gdrive/MyDrive/LULC_DCNN/Dataset/Training/training_features.npy')
y_train=np.load('/content/gdrive/MyDrive/LULC_DCNN/Dataset/Training/training_labels.npy')
X_test=np.load('/content/gdrive/MyDrive/LULC_DCNN/Dataset/Testing/testing_features.npy')
y_test=np.load('/content/gdrive/MyDrive/LULC_DCNN/Dataset/Testing/testing_labels.npy')


# In[16]:


# Training and testing dataset shape

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[17]:


# Training and testing data type
print(X_train.dtype, y_train.dtype, X_test.dtype, y_test.dtype)


# Now we will perform data normalization step. Data normalization ensures that input features are on a consistent scale, promoting stable training, faster convergence, and improved model generalization. It mitigates issues related to gradient descent, facilitates appropriate learning rate choices, and enhances the overall robustness of the model. 

# In[18]:


# Data Normalization of training and testing features
X_train=X_train/65535.0
X_test=X_test/65535.0


# Now we will perform one-hot encoding vectors. One-hot encoding transforms categorical labels into binary vectors, where each unique category corresponds to a binary value, enabling the model to better understand and differentiate between different classes during training and prediction.

# In[19]:


# Convert categorical labels into one-hot encoded vectors
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# # Building and Training the well-optimized CNN

# Now, this step is particularly crucial. We will design a well-optimized Convolutional Neural Network (CNN) architecture for classification, and proceed to train the model.

# In[20]:


# Training the Deep Convolutional Neural Network (DCNN)
# DCNN architecture

learning_rate=0.0001
Adm =Adam(learning_rate)
SIZE=5
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(2,2),strides=(1,1),padding='same', activation="relu", input_shape=(SIZE,SIZE,4)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2,2),strides=(1,1),padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(2,2),strides=(1,1),padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation="softmax"))
model.summary()

model.compile(optimizer=Adm, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=300,verbose=1, validation_data=(X_test, y_test), batch_size=128)


# # Save Trained CNN model in directory

# As we have trained the CNN model, we will now save it in our directory. Since we are performing classification in Google Colab, we will save our trained model in Google Drive

# In[ ]:


# Save trained DCNN model
model.save('/content/gdrive/MyDrive/LULC_DCNN/4bands_DCNN_LULCclassification.h5')


# # Load saved CNN model

# In[22]:


# Load saved trained DCNN model
from keras.models import load_model
model_load = load_model('/content/gdrive/MyDrive/LULC_DCNN/4bands_DCNN_LULCclassification.h5')


# Let's generate confusion matrices for the training and testing datasets. To do this, we'll first convert the training and testing labels into class indices. Converting back to class indices simplifies the comparison of model predictions with ground truth labels during evaluation or analysis. This enables a direct comparison between the predicted class (an index) and the true class label, facilitating the calculation of metrics such as accuracy, precision, and recall. Such conversion is particularly valuable when interpreting and presenting classification results

# In[23]:


#Training data confusion matrix
# convert the one-hot encoded labels (y_train) into class indices
train_labels=np.argmax(y_train, axis=1)
predict_x=model_load.predict(X_train)
classes_x=np.argmax(predict_x,axis=1)


# In[25]:


# import necessary libraries for training data confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Training data confusion matrix

confusion_matrix=confusion_matrix(train_labels,classes_x)
df_cm = pd.DataFrame(confusion_matrix,index = [ 'barrenland', 'builtup', 'fallowland','vegetation','wetland'],
                                                         columns = [ 'barrenland', 'builtup', 'fallowland','vegetation','wetland'])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm,annot=True,fmt="d",cmap='Blues')
plt.title('4bandsCNNModel_trainingdata_ConfusionMatrix', fontsize =15) # title with fontsize 20
plt.xlabel('predicted Label', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('True Label', fontsize = 15) # y-axis label with fontsize 15


# In[26]:


#Testing data confusion matrix
# converts the one-hot encoded labels (y_test) into class indices
test_labels=np.argmax(y_test, axis=1)
predict_x=model_load.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)


# In[29]:


# import necessary libraries for testing data confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Testing data confusion matrix

confusion_matrix=confusion_matrix(test_labels,classes_x)

df_cm = pd.DataFrame(confusion_matrix,index = [ 'barrenland', 'builtup', 'fallowland','vegetation','wetland'],
                                                         columns = [ 'barrenland', 'builtup', 'fallowland','vegetation','wetland'])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm,annot=True,fmt="d",cmap='Blues')
plt.title('4bandsCNNModel_testingdata_ConfusionMatrix', fontsize =15) # title with fontsize 20
plt.xlabel('predicted Label', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('True Label', fontsize = 15) # y-axis label with fontsize 15


# # Image Prediction with Trained CNN Model

# Now, we will install the 'pyrsgis' library and load the Sentinel-2 multispectral image for classification using the trained CNN model

# In[ ]:


# install pyrsgis library
pip install pyrsgis


# In[31]:


from pyrsgis import raster
import pyrsgis
import math


# In[32]:


# Generate image patches in the back-end. This part is modified from https://towardsdatascience.com/is-cnn-equally-shiny-on-mid-resolution-satellite-data-9e24e68f0c08
ds, predict_raster = raster.read('/content/gdrive/MyDrive/LULC_DCNN/sentinel2image_Classification.tif')
def CNNdataGenerator(mxBands, kSize):
    mxBands = mxBands /65535.0
    nBands, rows, cols = mxBands.shape
    margin = math.floor(kSize/2)
    mxBands = np.pad(mxBands, margin, mode='constant')[2:-2, :, :]

    features = np.empty((rows*cols, kSize, kSize, nBands))

    n = 0
    for row in range(margin, rows+margin):
        for col in range(margin, cols+margin):
            feat = mxBands[:, row-margin:row+margin+1, col-margin:col+margin+1]

            b1, b2, b3, b4 = feat
            feat = np.dstack((b1, b2, b3, b4))

            features[n, :, :, :] = feat
            n += 1

    return(features)

# Call the function to generate features tensor
new_features = CNNdataGenerator( predict_raster, kSize=5)
print(new_features.shape)
print('Shape of the new features', new_features.shape)


# In[33]:


# Predict new data
prediction= model_load .predict (new_features)


# In[34]:


# Converting predicted probabilities to class indices, The model's output, typically in the form of predicted probabilities for each class, needs to be converted into actual class predictions. argmax(axis=1) helps identify the index of the class with the highest predicted probability for each input sample.
pred = prediction.argmax(axis=1)


# # Export the predicted classified raster 

# In[35]:


prediction = np.reshape(pred, (ds.RasterYSize, ds.RasterXSize))

outFile = '/content/gdrive/MyDrive/LULC_DCNN/sentinelimage_Classification.tif'
raster.export(prediction, ds, filename=outFile, dtype='float')


# # Let's visualize it

# I think this time, instead of using 'pyrsgis,' we should explore another geospatial library, 'Rasterio,' for the visualization of rasters.

# In[36]:


# Let's print shape of multipspectral raster before visualization
print(predict_raster.shape)


# In[48]:


pip install rasterio


# In[50]:


# Take our multispectral raster and reshape into long 2d array (nrow * ncol, nband) for visualization using Rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
reshaped_img = reshape_as_image(predict_raster)
print(reshaped_img.shape)


# # Visualize multispectral and predicted classified raster

# In[60]:


# import necessary libraries for visualization
import rasterio
from rasterio.plot import adjust_band
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image

# Function for visualizing the original raster
def color_stretch(image, index):
    colors = image[:, :, index].astype(np.float64)
    for b in range(colors.shape[2]):
        colors[:, :, b] = adjust_band(colors[:, :, b])
    return colors

# Create colormap
clist = ["MediumTurquoise", "red", "Wheat", "LimeGreen", "Blue"]
cmap = matplotlib.colors.ListedColormap(clist)

# Define class names
class_names = ["Barren land", "Urban", "Fallow land", "Agriculture", "Wet land"]

# Create legend patches using class names
legend_patches = [mpatches.Patch(color=cmap(i), label=class_names[i]) for i in range(len(clist))]

# Visualize the original raster and the classified image
fig, axs = plt.subplots(1, 2, figsize=(20, 30))

img_stretched = color_stretch(reshaped_img, [3, 2, 1])
axs[0].imshow(img_stretched)
axs[0].set_axis_off()

axs[1].imshow(prediction, cmap=cmap, interpolation='none')
axs[1].set_axis_off()
axs[1].legend(handles=legend_patches, title='LULC_Classes', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

