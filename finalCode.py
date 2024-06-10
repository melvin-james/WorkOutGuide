import os
import cv2
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import os
import cv2
import numpy as np

def load_images_from_directory(data_dir, labels, image_size):
    data = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(img, image_size)
                data.append([resized_img, class_num])
            except Exception as e:
                print(f"Error loading image: {img_path}")
                print(e)
    return tuple(data)

data_folder = r'D:\mp\newdataset'

for dirpath, dirnames, filenames in os.walk(data_folder):
    # Count the number of image files in each subfolder
    image_count = sum(1 for filename in filenames if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')))

    # Display the subfolder path and the corresponding image count
    print(f"Subfolder: {dirpath}")
    print(f"Number of Images: {image_count}")
    print()

# Set the directory path and the list of labels
data_dir = r'D:\mp\newdataset'
labels = ['curl', 'pushup', 'situp', 'squats']

# Specify the desired image size
image_size = (224, 224)

# Load the images from the directory
dataset = load_images_from_directory(data_folder, labels, image_size)

# Print the shape of the dataset
print("Dataset shape:", len(dataset))

import os
#no. of images in each path of root folder

data_folder = r'D:\mp\newdataset'

for dirpath, dirnames, filenames in os.walk(data_folder):
    # Count the number of image files in each subfolder
    image_count = sum(1 for filename in filenames if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')))

    # Display the subfolder path and the corresponding image count
    print(f"Subfolder: {dirpath}")
    print(f"Number of Images: {image_count}")
    print()

import numpy as np
from sklearn.model_selection import train_test_split

# Split the dataset into train, test, and validation sets
train_size = 0.7
test_size = 0.2
val_size = 0.1

# Split the dataset into train and remaining (test + validation)
train, remaining_data = train_test_split(dataset, train_size=train_size, random_state=42)

# Split the remaining data into test and validation
test, val = train_test_split(remaining_data, train_size=test_size/(test_size+val_size), random_state=42)

# Convert lists to numpy arrays
train = np.array(train)
test = np.array(test)
val = np.array(val)

# Print the shapes of the resulting datasets
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
print("Validation data shape:", val.shape)

# Split the data into features and labels

x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []


for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
print(len(x_val))
print(len(y_val))

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) /255
x_test = np.array(x_test) / 255

IMAGE_SIZE = 224 

# Resize data for deep learning
x_train = x_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y_test = np.array(y_test)

y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)
#model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers
for layer in vgg16_model.layers:
    layer.trainable = False

# Create a new model based on VGG16
model = Sequential()
model.add(vgg16_model)

# Add fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=12, batch_size=32, callbacks=[reduce_lr])

print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)  # Convert predictions to class labels
true_labels = np.argmax(y_test, axis=1)  # Convert true labels to class labels

print(classification_report(true_labels, predicted_labels, target_names=labels))

cm = confusion_matrix(true_labels, predicted_labels)
cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(8,8))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=labels, yticklabels=labels)
plt.show()

y_pred = model.predict(x_test)
num_classes = 4

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves for each class
plt.figure(figsize=(12, 12))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (area = %0.2f)' % roc_auc["micro"], linestyle=':', linewidth=4)

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Show the plot
plt.show()

model.save("Model with VGG16, (91.5% acc).h5")

import os

# Get the current working directory
current_directory = os.getcwd()

# Specify the name of the saved model file
model_file = "Model with VGG16, (91.5% acc).h5"

# Join the current directory with the model file name to get the full path
model_path = os.path.join(current_directory, model_file)

# Print the path of the saved model
print("Model path:", model_path)

source = r'C:\Users\91813\Model with VGG16, (91.5% acc).h5'
destination = 'D:\mp'

# Move the file to the new location
shutil.move(source, destination)

correct = np.nonzero(predicted_labels == true_labels)[0]
incorrect = np.nonzero(predicted_labels != true_labels)[0]
print("No. of correctly predicted images: ", len(correct))
print("No. of incorrectly predicted images: ", len(incorrect))

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
fig.suptitle('Correctly Classified Images', fontsize=16)
for i, idx in enumerate(correct[:4]):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(x_test[idx])
    axes[row, col].set_title("Predicted Class: {}\nTrue Class: {}".format(labels[predicted_labels[idx]], labels[true_labels[idx]]))
    axes[row, col].axis('off')
plt.tight_layout()
plt.show((cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)))

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
fig.suptitle('Incorrectly Classified Images', fontsize=16)
for i, idx in enumerate(incorrect[:4]):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(x_test[idx])
    axes[row, col].set_title("Predicted Class: {}\nTrue Class: {}".format(labels[predicted_labels[idx]], labels[true_labels[idx]]))
    axes[row, col].axis('off')
plt.tight_layout()
plt.show((cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)))

from tensorflow.keras.models import load_model

# Load the pre-trained model from a file
model = load_model('D:\mp\Model with VGG16, (91.5% acc).h5')

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the image
image_path = r'D:\mp\newdataset\PUSHUP\bodybuilder-doing-push-ups-royalty-free-image-1636040776_augmented_2.png'
image = Image.open(image_path)

# Resize the image to match the input shape of VGG16
image = image.resize((224, 224))

# Convert the image to a numpy array
image_array = np.array(image)

# Expand dimensions to create a batch of size 1
image_array = np.expand_dims(image_array, axis=0)

# Preprocess the image
preprocessed_image = preprocess_input(image_array)

# Perform inference on the preprocessed image using the loaded model
predictions = model.predict(preprocessed_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Define the mapping of class indices to labels
class_labels = ['curl', 'pushup', 'situp', 'squats']

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print('Predicted class:', predicted_class_label)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the test accuracy
print('Test accuracy:', test_accuracy)

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the test image
test_image_path = r'D:\mp\newdataset\SQUATS\30-day-squat-challenge-lead-1588712943_augmented_0_augmented_0.png'
test_image = Image.open(test_image_path)

# Resize the image to match the input shape of VGG16
test_image = test_image.resize((224, 224))

# Convert the image to a numpy array
test_image_array = np.array(test_image)

# Expand dimensions to create a batch of size 1
test_image_array = np.expand_dims(test_image_array, axis=0)

# Preprocess the image
preprocessed_image = preprocess_input(test_image_array)

# Perform inference on the preprocessed image using the loaded model
predictions = model.predict(preprocessed_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Define a list of class names in the same order as the model's output classes
class_names = ['CURL', 'PUSHUP', 'SITUP', 'SQUATS']

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Print the predicted class name
print('Predicted class name:', predicted_class_name)
