import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as image
import sklearn.model_selection
import tensorflow.keras.optimizers

input_folder = "all_letters_image/all_letters_image"
all_letters_filename = os.listdir(input_folder)

i = Image.open(input_folder + "/07_201.png")
i_arr = np.array(i)

def img_to_array(img_name, input_folder):
    img = image.load_img(input_folder + '/' + img_name, target_size=(32,32))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def data_to_tensor(img_names, input_folder):
    list_of_tensors = [img_to_array(img_name, input_folder) for img_name in img_names]
    return np.vstack(list_of_tensors)



data = pd.read_csv("all_letters_info.csv")
image_names = data['file']
letters = data['letter']
targets = data['label'].values
tensors = data_to_tensor(image_names, input_folder)

X = tensors.astype("float32")/255
y = targets

img_rows, img_cols = 32, 32
num_classes = 33

y = keras.utils.to_categorical(y-1, num_classes)

X_train_whole, X_test, y_train_whole, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train_whole, y_train_whole, test_size=0.1, random_state=1)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample/image mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

print(X_train)


deep_RU_model = keras.models.Sequential()

deep_RU_model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (img_rows,img_cols,3)))
deep_RU_model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
deep_RU_model.add(keras.layers.Dropout(0.25))


deep_RU_model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

deep_RU_model.add(keras.layers.Dropout(0.25))


deep_RU_model.add(keras.layers.Flatten())
deep_RU_model.add(keras.layers.Dense(256, activation = "relu"))
deep_RU_model.add(keras.layers.Dropout(0.5))
deep_RU_model.add(keras.layers.Dense(33, activation = "softmax"))

optimizer = tensorflow.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

deep_RU_model.compile(loss="categorical_crossentropy", optimizer = optimizer,metrics=["accuracy"])

learning_rate_reduction = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
mc = tensorflow.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = deep_RU_model.fit(datagen.flow(X_train,y_train, batch_size=90), validation_data = (X_val, y_val),
                            epochs=139, callbacks=[learning_rate_reduction, es, mc])

deep_RU_model.save('best_model.h5')
