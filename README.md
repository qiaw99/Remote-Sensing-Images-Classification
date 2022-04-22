# Remote-Sensing-Images-Classification

## Installation
```
git-lfs
```

## Environment
Google Colab with GPU, TensorFlow version: 2.8.0

## Project Structure
* [Data](#40)
* [checkpoints](#41)
* [data-in-numpy](#42)
* [*.ipynb](#43)

## How to use it? 
### 1. Get data: 
1. Initially, I only get raw data in zip file(which you can find it in folder **data**) downloaded from Google Drive. You should extract zip file as one folder and should have the following file-structure:
    ```
    - Challenge_dataset
        - train
            - agricultural
                - agricultural00.tif
                - ...
            - airplane
                - ...
            - ...
        - test
            - agricultural
                - agricultural00.tif
                - ...
            - airplane
                - ...
            - ...
    ```
    >  TODO(how preprocessing)

2. Alternatively, you could use data that I've preprocessed which is located in folder **data-in-numpy**. In it, the structure is as following: 
    ```
    - test.npy
    - train.npy
    ```
    You should put them into folder **Challenge_dataset**. The training data has size: **(1076, 256, 256, 3)** and testing data has size **(276, 256, 256, 3)**. Then you could reload the data with help of two npy files: 
    ```py
    with open("test.npy", "rb") as f:
        X_val = np.load(f)
        y_val = np.load(f)

    with open("train.npy", "rb") as f:
        X = np.load(f)
        y = np.load(f)
    ```
    However, this data cannot be directly used for training and evaluation, you should normalize them firstly before use: 
    ```py
    # Normalize pixel values to be between 0 and 1
    X = X / 255.0
    ```
With that, we can virtualize some of images: 
```py
plt.figure(figsize=(10,10))
for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i*40])
    plt.xlabel(class_names[y[i*40]])
plt.show()
```
![](./img/data.png)
### 2. Build and train the model
1. model.ipynb
![](./img/loss.png)
![](./img/loss1.png)

2. Use the train model: 
    All weights are stored in folder **checkpoints**. You should put this entire folder into **Challenge_dataset**. 
    * Build a model(details will be explained afterwards). 
```py
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

names[-1] # getting the name of the last conv layer

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
```
Then goto the correct parrent folder:
```py
os.chdir("drive/MyDrive/Challenge_dataset/")
```
Then you could reload weights for the model: 
```py
model.load_weights('./checkpoints/my_checkpoint')
```
With this model, you could now to predict the image class like following: 
```py
classif_prob = model.predict(X_val)
pred_classes_argmax = np.argmax(classif_prob,axis=-1)
predicted_cls = pred_classes_argmax[0]
print("Predicted class:", predicted_cls)

> Predicted class: 0
```
3. Evaluation
    The pretrained model cannot be directly used for evaluation, but rather firstly compile them: 
```py
from tensorflow.keras.optimizers import RMSprop

# compile the model
model.compile(optimizer=RMSprop(lr=0.001),
            loss='SparseCategoricalCrossentropy',
            metrics=['accuracy'])
```
After that, you can try to evaluate it: 
```py
values = model.evaluate(X_val, y_val)
print("{}:{},{}:{}.".format(model.metrics_names[0],values[0],model.metrics_names[1],values[1]))

> 9/9 [==============================] - 59s 6s/step - loss: 3.2318 - accuracy: 0.4167
loss:3.2318239212036133,accuracy:0.4166666567325592.
```


##  Introduction
This is a **Coding Challenge from Project CV4RS at TU Berlin.** Data consists of 1091 training data and 279 testing data which are from 21 different classes:
```py
class_names = [
'agricultural', 'airplane',                  'baseballdiamond', 'beach', 
'buildings','chaparral', 
'denseresidential', 'forest', 
'freeway', 'golfcourse',
'harbor', 'intersection', 
'mediumresidential', 'mobilehomepark',
'overpass', 'parkinglot', 
'river', 'runway', 'sparseresidential',
'storagetanks', 'tenniscourt']
```
However, during preprossing step, it was figured out that not all data can be used. Most of data have shape: (256, 256, 3). But some of data have different size and I would just discard them.

