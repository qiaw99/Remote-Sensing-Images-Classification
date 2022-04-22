import tensorflow as tf
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten, ReLU, GlobalAveragePooling2D, Input

def evaluate():
    class_names = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
                'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
                'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
                'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                'storagetanks', 'tenniscourt']
    
    print("[Info] Restoring model from checkpoint.")
    
    # Instantiate the ResNet50 model
    conv_base =tf.keras.applications.resnet.ResNet50(input_shape=(256, 256, 3),
                            include_top=False,
                            weights='imagenet')

    # freeze the layers
    for layer in conv_base.layers:
        layer.trainable = False

    names = []
    for layer in conv_base.layers:
        names.append(layer.name)

    # names[-1] # getting the name of the last conv layer

    last_layer = conv_base.get_layer('conv5_block3_out')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    inputs = Input(shape = (256, 256, 3))
    x = conv_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)
    # Add a final softmax layer for classification
    output = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs, output)

    print("[Info] loading weights.")
    model.load_weights('./tmp/folder/my_checkpoint')
    with open("./data-in-numpy/test.npy", "rb") as f:
        X_val = np.load(f, allow_pickle=True)
        y_val = np.load(f, allow_pickle=True)

    X_val = X_val / 255.0

    classif_prob = model.predict(X_val)
    pred_classes_argmax = np.argmax(classif_prob,axis=-1)

    predicted_cls = pred_classes_argmax[0]
    print("Predicted class:", predicted_cls)

    # compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='SparseCategoricalCrossentropy',
                metrics=['accuracy'])
    
    print("[Info] Evaluating the model.")

    values = model.evaluate(X_val, y_val)
    
    print("[Info] Results: {}:{},{}:{}.".format(model.metrics_names[0],values[0],model.metrics_names[1],values[1]))


if __name__ == "__main__":
    evaluate()