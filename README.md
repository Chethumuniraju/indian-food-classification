# Indian-Food-Classification

## Table of Contents
+ [About](#about)
+ [Exploratory Data Analysise](#exploratory-data-analysis)
+ [Model](#model)
+ [Training](#training)
+ [Results Evaluation](#results-evaluation)
+ [Conclusion](#conclusion)
+ [Reference](#reference)

## About

Indian food dataset is a labelled data set with different food classes. Each food class contains 1000s of images. Using the data provided, a deep learning model built on TensorFlow is trained to classify into various classes in dataset.

<br>**Epoches:** 200
<br>**Batch_size:** 32

Images are split to train and test set with 4020 images belonging to 80 different classes. 

## Exploratory Data Analysis

Let's preview some of the images.

<img src = "https://github.com/gnpaone/Indian-food-classification/blob/main/Pictures/EDA.png">

The size of the images are inconsistent, so all the images are scaled to same size, so we dont have to worry about inconsistency.

## Model
To create a convolution neural network to classfied the images, Sequential model is used.

```python
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256,activation = 'relu'),
    layers.Dense(n_classes,activation = 'softmax'),
])
```

## Training

<img src = "https://github.com/gnpaone/Indian-food-classification/blob/main/Pictures/training.png">

Model accuracy increased over each epoch, overfitting started at around 40 epochs. The model achieved validation accuracy of **92.94%** with a 0.2418 cross entropy validation loss.

## Results Evaluation

Preview some predictions from the model:

<p>First Image to Predict:</p>
<img src = "https://github.com/gnpaone/Indian-food-classification/blob/main/Pictures/test.png">
<p>Actual Label: imarti</p>
<p>Predicted Label: imarti</p>

Now, let's examine in more detail how the model performs and evaluate those predictions.

<img src = "https://github.com/gnpaone/Indian-food-classification/blob/main/Pictures/model.png">

## Conclusion

With the given data sets for 80 classes of Indian food, the model final accuracy reached 92.94% with cross entropy validation loss of 0.2418.

## Reference

[Indian food Dataset](Dataset/Indian-Food-Images)
- [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)

<p align="center">
<b>⭐ Please consider starring this repository if it helped you! ⭐</b>
</p>

