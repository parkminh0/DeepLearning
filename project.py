#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense


# In[ ]:


labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
train_img = []
train_label = []
test_img  = []
test_label = []

image_size=150

for label in labels:
    trainPath = os.path.join(r"C:\Users\minhopark\Desktop\ex\dataset\train",label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file))
        image = cv2.resize(image, (image_size, image_size))
        train_img.append(image)
        train_label.append(label)
        
    testPath = os.path.join(r"C:\Users\minhopark\Desktop\ex\dataset\test",label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file))
        image = cv2.resize(image, (image_size, image_size))
        test_img.append(image)
        test_label.append(label)


# In[4]:


plt.figure(figsize = (15,5));
lis = ['Train', 'Test']
for i,j in enumerate([train_label, test_label]):
    plt.subplot(1,2, i+1);
    sns.countplot(x = j);
    plt.xlabel(lis[i])


# In[5]:


k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample From Each Label',size=18,fontweight='bold',
             fontname='monospace',y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if train_label[j]==i:
            ax[k].imshow(train_img[j])
            ax[k].set_title(train_label[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1


# In[6]:


train_img, val_img, train_label, val_label = train_test_split(train_img, train_label, test_size=0.22, random_state=28)


# In[32]:


train_img.shape, val_img.shape


# In[33]:


train_label.shape, val_label.shape


# In[7]:


train_img


# In[8]:


train_label


# In[9]:


train_img, val_img, test_img = np.array(train_img), np.array(val_img), np.array(test_img)
train_label, val_label, test_label = np.array(train_label), np.array(val_label), np.array(test_label)


# In[10]:


train_img, val_img, test_img = train_img/255, val_img/255, test_img/255


# In[11]:


train_img


# In[12]:


train_label


# In[13]:


#Label encoding & One-hot encoding
train_label_enc = []
for i in train_label:
    train_label_enc.append(labels.index(i))
train_label = train_label_enc
train_label = tf.keras.utils.to_categorical(train_label)

val_label_enc = []
for i in val_label:
    val_label_enc.append(labels.index(i))
val_label = val_label_enc
val_label = tf.keras.utils.to_categorical(val_label)

test_label_enc = []
for i in test_label:
    test_label_enc.append(labels.index(i))
test_label = test_label_enc
test_label = tf.keras.utils.to_categorical(test_label)


# In[14]:


train_label


# In[15]:


class CNN(tf.keras.Model):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.drop1= tf.keras.layers.Dropout(0.25)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation = 'relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.drop2 = tf.keras.layers.Dropout(0.25)
        
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation = 'relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.drop3 = tf.keras.layers.Dropout(0.25)

        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation = 'relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.drop4 = tf.keras.layers.Dropout(0.3)
        
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation = 'relu')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.drop5 = tf.keras.layers.Dropout(0.3)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.drop6 = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.drop4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.drop5(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.drop6(x)
        return self.out(x)


# In[16]:


def get_model():
    return CNN()


# In[17]:


model = get_model()


# In[18]:


model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'])


# In[19]:


reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


# In[20]:


history = model.fit(train_img, train_label, batch_size=40,
                    validation_data=(val_img, val_label),
                    epochs=25,
                    callbacks=[reduce_lr])


# In[23]:


model.summary()


# In[21]:


val_loss, val_acc = model.evaluate(val_img,val_label)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")


# In[23]:


plt.style.use("ggplot")
plt.figure(figsize=(12,6))
epochs = range(1,26)
plt.subplot(1,2,1)
plt.plot(epochs,history.history["accuracy"],'go-')
plt.plot(epochs,history.history["val_accuracy"],'ro-')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Val'],loc = "upper left")

plt.subplot(1,2,2)
plt.plot(epochs,history.history["loss"],'go-')
plt.plot(epochs,history.history["val_loss"],'ro-')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Val'],loc = "upper left")

plt.show()


# In[24]:


predict = model.predict(test_img)
predict = np.argmax(predict,axis=1)
pred_label = np.argmax(test_label,axis=1)


# In[27]:


accuracy = np.sum(predict==pred_label)/len(predict)
print("Accuracy on testing dataset: {:.2f}%".format(accuracy*100))


# In[29]:


plt.figure(figsize=(12,9))
for i in range(10):
    pred_res = "Correctly predicted!"
    sample_idx = random.choice(range(len(test_img)))
    plt.subplot(2,5,i+1)
    plt.imshow(test_img[sample_idx])
    if predict[sample_idx] != pred_label[sample_idx]:
        pred_res = "Mispredicted!"
    plt.xlabel(f"Real type: {pred_label[sample_idx]}\n Predicted: {predict[sample_idx]}\n {pred_res}")
    
plt.tight_layout()
plt.show()


# In[ ]:
