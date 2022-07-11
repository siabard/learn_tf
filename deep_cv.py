## Convolutional neural network
# Using CIFAR image Daaset

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## image 
# IMG_INDEX = 1
# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.library)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

# CNN archtecture
# stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers
model = models.Sequential()

# layer 1 : input shape of data will be 32, 32,3 and we will process 32 filters of size 3x3 over input data
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
# layer 2 : max pooling operation using 2x2 samples and a stride of 2
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# add dense layer
# classify layer

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Training
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluating
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)