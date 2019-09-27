# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (30, 30, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding 3rd convolution layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3,3), activation = 'relu'))

classifier.add(Conv2D(128, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#classifier.add(Conv2D(512, (3,3), activation = 'relu'))
#classifier.add(Conv2D(512, (3,3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(units = 512, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 46, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32

training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size = (30, 30),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Dataset/Test',
                                            target_size = (30, 30),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

# start training
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 78200//batch_size,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 13800//batch_size)


classifier.save('classifier.h5')

# check classification mapping
training_set.class_indices

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Dataset/test1.jpg', target_size = (30, 30))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = np.argmax(classifier.predict(test_image))
