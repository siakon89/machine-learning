from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import optimizers
from tensorflow.contrib.eager.python import tfe

import tensorflow as tf
import os
import argparse
import numpy as np


def get_data(given_dir, file_names):
    
    x_name = file_names[0]
    y_name = file_names[1]
    
    x_data = np.load(os.path.join(given_dir, x_name))
    y_data = np.load(os.path.join(given_dir, y_name))
    print(x_name, x_data.shape,y_name, y_data.shape)

    return x_data, y_data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))
    
    # Load the data
    train_images, train_labels = get_data(args.train, ['train_images.npy', 'train_labels.npy'])
    test_images, test_labels = get_data(args.test, ['test_images.npy', 'test_labels.npy'])
    
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=train_images[0].shape))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=adam, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(
        train_images, 
        train_labels, 
        validation_data=(test_images, test_labels), 
        epochs=epochs,
        batch_size=batch_size
    )
    
    # evaluate on test set
    scores = model.evaluate(test_images, test_labels, batch_size, verbose=2)
    print("Test MSE :", scores)
    
    # create a separate SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    