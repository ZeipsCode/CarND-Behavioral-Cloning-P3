# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with only six layers. First i added a Lambda Layer to normalize the input data to a range between 0 and 1. The second layer crops the image to a defined region of interest between the 70th and 25th row of the input image. By doing this i essentially crop out the sky and the hood of the car. Then comes to main part of the model: one convolutional layer with a filter size of 3x3 and depth of 6 and a relu nonlinearity activation.Then the data are passed to a maxpooling layer with a pooling size of 3x3 which is followed by a dropout layer with a dropout probability of 25 % to prevent overfitting. The last layer in my model is a fully connected layer with only 1 neuron which determines the output steering angle. I am using a tanh activation function for the fully connected layer to get values between -1 and 1.  (model.py lines 56 - 63)


#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Essentially i used only the included data and did not create my own dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because it is very good at recognizing shapes in the mnist dataset. So if figured it would be doing ok if it was used to recognize lane lines. 
It did pretty well on the test and validation set and could actually drive the car around most of the track.

After that i tried the much deeper Nvidia-net introduced in class. But this model turned out to be way too complex and harder to train. In my limited testing it did way worse than the LeNet architecture so i figured i could try and simplify my architecture as much as possible. That way i came up with my architecture of only six layers which works ok on the simulator test track and does not leave the road at any point.


#### 2. Creation of the Training Set & Training Process

I only used the included dataset in the Udacity repository to train my network. But i did some augmentation on it by flipping the data, here is an example of one image before and after flipping:

![alt text][image6]
![alt text][image7]


After augmentation i had 22180 images which i then shuffled for training and put 20 % of it on the validation set. It turned out that training the model for 10 Epochs was enough to get the model to converge so that the validation accuracy did not fall further from that point on.
