# pretrained.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
keras = tf.keras

import tensorflow_datasets as tfds 
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True,)
get_label_name = metadata.features['label'].int2str # function object to get labels

# sample view
# for image, label in raw_train.take(2):
#    plt.figure()
#    plt.imshow(image)
#    plt.title(get_label_name(labe))
# plt.show()

## Data preprocessing

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Picking a pretrained model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# for image, _ in train_batches.take(1):
#    pass
# feature_batch = base_model(image)
# print(feature_batch.shape)

# freezing base
base_model.trainable = False

## Adding classfifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

## Train model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

# evaluate model
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# train model
history = model.fit(train_batches, epochs = initial_epochs, validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')