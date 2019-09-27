# -*- coding: utf-8 -*-

# import libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


class MyCnnModel():

    # import and pre-process data
    def preprocess_data(self):
        self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = False,
                                           )    
        test_datagen = ImageDataGenerator(rescale = 1./255)        
        self.batch_size = 32
        self.training_set = self.train_datagen.flow_from_directory('Dataset/Train',
                                                         target_size = (32, 32),
                                                         batch_size = self.batch_size,
                                                         class_mode = 'categorical')    
        self.test_set = test_datagen.flow_from_directory('Dataset/Test',
                                                    target_size = (32, 32),
                                                    batch_size = self.batch_size,
                                                    class_mode = 'categorical')
        self.training_samples = self.training_set.samples
        self.test_samples = self.test_set.samples
          
    # create CNN model
    def build_cnn(self):
        self.classifier = Sequential()
        self.classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
        self.classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Conv2D(filters = 128, kernel_size= (2 ,2), activation = 'relu'))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dropout(0.5, seed = 9))
        self.classifier.add(Dense(units = 46, activation = 'softmax'))
        self.classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.classifier.summary()
        
    # start training
    def train_cnn(self):
        self.history = self.classifier.fit_generator(self.training_set,
                                 steps_per_epoch = self.training_samples//self.batch_size,
                                 epochs = 1,
                                 validation_data = self.test_set,
                                 validation_steps = self.test_samples//self.batch_size)
        print (self.history)

    # Save model
    def save_model(self):
        self.classifier.save('DevanagariHandwrittenCharacterDataset_' + str(self.history.history['val_acc']))

# let's begin
def main():
    mymodel = MyCnnModel()        
    mymodel.preprocess_data()
    mymodel.build_cnn()
    mymodel.train_cnn()
    mymodel.save_model()
    
if (__name__ == '__main__'):
    main()

        
        


        
        
        
        
        
        
        
        