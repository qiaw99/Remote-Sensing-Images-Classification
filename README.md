# Remote-Sensing-Images-Classification

## Installation
```
git-lfs clone git@github.com:qiaw99/Remote-Sensing-Images-Classification.git
```
*Notice that git might not be enough, because data stored in \*.npy and \*.zip files exceed the limitation of git(100MB). So please instead using [git-lfs](https://git-lfs.github.com/).*

## Environment
Windows, Google Colab with GPU, TensorFlow version: 2.8.0

## Overview
* Report
    * [Project structure](#44)
    * [Getting data](#40)
    * [Experimental setup and baseline](#41)
    * [Evaluation](#43)

<h2 id=44> The whole project structure should be like this: </h2>

```
- Challenge_dataset
    - train
    - test
    - train.npy
    - test.npy
    - checkpoints
        - checkpoint
        - my_checkpoint.data-00000-of-00001
        - my_checkpoint.index
```

## Evaluation of model

<div>
  <a href="https://github.com/qiaw99/Remote-Sensing-Images-Classification/actions"><img src="https://github.com/wkentaro/gdown/workflows/ci/badge.svg"></a>
</div>

If you just want to see how good this model works, you can check it by [pipeline](https://github.com/qiaw99/Remote-Sensing-Images-Classification/actions). However, if you would like to test the model on your own data, you could use __**evaluation.ipynb**__ and replace the corresponding *.npy files. If you want to do it convenientlly, we can execute __code_challenge.ipynb__ but you have to structure your project like [this](#44). The training happens in __code_challenge.ipynb__, so you could see all the details there. In __model.ipynb__, only the process is provided but the model has not been trained.  *__Note: do not use evaluate.py for evaluation. This is only for pipeline!__* 

# Pipeline:
## Testing environment: 
Ubuntu 18.04, Tensorflow 2.8.0.

**Problem:**
I met the problem that in pipeline, __*.npy__ files cannot be correctly read, see [issue here](https://github.com/qiaw99/Remote-Sensing-Images-Classification/issues/1). The solution is, by using package [gdown](https://github.com/wkentaro/gdown) to download them directly from Google Drive:
``` yml
gdown folder_address -O /tmp/folder --folder
gdown file_id
```


#  Introduction
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

# Report 
<h2 id=40> 1. Get data:</h2> 

1. Initially, I only get raw data in zip file(which you can find it in folder **data**) downloaded from [Google Drive](https://drive.google.com/file/d/1zVkU9eMuerAJ_lbC2Uj8mAcn6rueuAK7/view?usp=sharing). You should extract zip file as one folder and should have the following file-structure:
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
    * __Preprocessing image data:__ 
```py

import cv2 as cv
from skimage import io

X = []
y = []
error = []
counter = 0

for c in class_names:
  os.chdir(os.getcwd() + "/" + c)
  print(os.getcwd())
  files = os.listdir(os.getcwd()) 
  print(files)
  for file in files: 
    img = cv.imread(file)
    if(img.shape == (256, 256, 3)):
      X.append(img)
      y.append(counter)
    else:
      error.append(c + file)
  os.chdir(os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))
  print(os.getcwd())
  counter += 1
```
We can read all images with help of opencv and store those images as numpy arrays in shape of (256, 256, 3). As we go through all subfolders of **train**, we  should also annotate the corresponding label to each image using index of range from 0 to 20 standing for each class as mentioned before.

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

 However, this data cannot be directly used for training and evaluation, you should normalize them firstly before use: 
```py
# Normalize pixel values to be between 0 and 1
X = X / 255.0
```

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

X, y are all data for training which we have to split them into traing data and validation data(80/20%) before we can really start training our model. In this case, I would use the defined function from sklearn: 
```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
<h2 id=41>2. Experimental setup and baseline </h2>

### __2 a)__
Now, we should look into the data itself. 
![](./img/Figure_1.png)

The x-axis stands for each corresponding class and y-axis stands for the number of images contained in each subfolders. Observe that the data is not so equally distributed. The average number of images is __51.95__ as in the figure listed. Furthermore, we see that the variance of data is quite large which means __the data distribution is not balanced!__

### __2 b)__
The exact model will be discussed in the next part. 

### __2 c)__ 
A baseline model should be established. However, I only build one model because of time limitation. But I think my model should be in some sense reasonably good to do the classification. Nevertheless, I would briefly explain my initial idea if I would implement a model for classfication task. The first choice to handle image problems is of course using __Convolutional Neural Network(CNN)__ by executing a series of convolutions and pooling operations which is ended by a set of __fully connected layers__ for classifications. However, we might encounter problems like vanishing or exploding gradients. Provided that the CNN is relatively shallow,  we may use __Batch Normalization__ or __Layer Normalization__ to fix it. But what if we just keep stacking more and more convolutional and pooling layers? With the increasing number of layers, the performance(accuracy) gets saturated which is not caused by overfitting. Since the given classification problem consists of 21 classes, I decided to use __ResNet 50__ which was in 2015 proposed by __Kaiming et. al__. ResNet can solve this problem by stacking multiple identity blockes which refer to residual path and convolutional blocks. 

__Improvements:__
* Within model: ResNet uses already Batch Normalization. What's more, we could use __dropout__ and __data augmentation__ like __clipping, rotation__, etc. so that the model can become more stable to unseen images.
* For model: In recent year, __transformer__ using __Attention mechanism__ which was published in 2017 is proven as really well performance model in the field of __Natural Language Processing(NLP)__. Recently, transformer is used also in the field of Computer Vision as __vision transformer(ViT)__ which is published in 2020 by putting more information(patch and position embedding). Thus, I would believe that using ViT would be a good way to improve. 

### __2 d)__
The evaluation metrics for classification is commonly computing __the cross entropy loss between the labels and predictions__.

<h2 id=43> 3. Build and train the model</h2>

1. model.ipynb


2. Use the trained model from me: 
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
I've decided that for convolutional part, the pretrained ResNet50 will be used which we can download it from keras application. For transfer learning, we should freeze all layers contained in ResNet50. After the last layer from ResNet, we have to do __GlobalAveragePooling__ once and flatten the output array so that we can pass them to __fully connected layers__ which are consist of 512 units layer and 21 units layer for classification. 

### __Hyperparameters:__
- Number of epochs: 100
- Learning rate: 0.001

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

We can plot some images to check the true labels and the predicted labels: 
```py
plt.figure(figsize=(20,20))
for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_val[i*10])
    plt.xlabel("True label: " + class_names[y_val[i*10]] + ", prediced: " + class_names[pred_classes_argmax[i*10]])
plt.show()
```
 <img src="./img/predicted.png" width = "1000" height = "500"  align=center />
 
It's a little bit unclear, but you could find this image in **img/predicted.png.**

![](./img/loss.png)
![](./img/loss1.png)

From both figures we can observe that, the accuracy keeps increasing as epochs getting larger and in contrast, the loss keeps decreasing. You can see the final result in __code_challenge.ipynb__: 
```py
Epoch 100/100
27/27 [==============================] - 11s 394ms/step - loss: 0.8822 - accuracy: 0.7105 - val_loss: 1.4790 - val_accuracy: 0.6435
```
In the end, we get accuracy for training 71% and for validation 64% which is pretty good. The final test accuracy is only 41%, there is a huge gap between validation and testing accuracy which results that I could actually train the model with more iterations. However, I don't have that much time for it. [TODO] 


