#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/data_visualization_initial.png "Visualization of training data"
[image12]: ./images/data_visualization_valid.png "Visualization of validation data"
[image13]: ./images/data_visualization_test.png "Visualization of test data"
[image2]: ./images/orignal%20vs%20grayscale.png "Grayscaling"
[image9]: ./images/grayscale%20vs%20normalized.png "Normalization"
[image10]: ./images/data_visualization_more_data.png "Visualization After Adding More Data"
[image3]: ./images/extra%20data.png "Extra Data"
[image4]: ./sample-traffic-signals/1x.png "Traffic Sign 1"
[image5]: ./sample-traffic-signals/2x.png "Traffic Sign 2"
[image6]: ./sample-traffic-signals/3x.png "Traffic Sign 3"
[image7]: ./sample-traffic-signals/4x.png "Traffic Sign 4"
[image8]: ./sample-traffic-signals/5x.png "Traffic Sign 5"
[image11]: ./images/new_images_softmax_visualization.png "Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/reshavsingla/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for training data for each type of signal. The data varies a lot among the signal types.

![alt text][image1]

Here is the chart for the validation data.

![alt text][image12]

and this is the chart for the test data,

![alt text][image13]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps in reducing the training time as the dimensions are decreased by a factor of 3.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it decreases the variation in data points decreases significantly.Also  Different features could encompass far different ranges and a single learning rate might make some weights diverge.

![alt text][image9]

I decided to generate additional data because the data was very unevenly distributed and had a lot more data for couple of type of signals

To add more data to the the data set, I used the following techniques because they add variation to the image which still keep the signal recognizable and just change the perspective of the imgage.

Here is an example of an original image and augmented images:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:-
- The number of data points increases from 34799 to 68657.
- The data is much more evenly distributed for different traffic signals preventing the model to bias towards the one having more data.

![alt text][image10]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| input 5x5x16 , output 400 				|
| Fully connected		| input 400 , output 120		|
| RELU					|												|
| Dropout					|					50%							|
| Fully connected		| input 120 , output 84		|
| RELU					|												|
| Dropout					|					50%							|
| Fully connected		| input 84 , output 43		| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer and had following parameters:-
1) Batch size - 128
2) Epochs - 30
3) Learning rate - 0.00095
4) Drop rate - 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.2% 
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I have used the Lenet architecture as it is a good starting point for image recognition.

* What were some problems with the initial architecture?

The low accuracy 89-90% for validation data caused by over fitting (high accuracy on training data) of the initial archtecture had to be improved.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The architecture as adjusted by adding 2 dropout layers to improve accuracy and decrease over fitting. I tried improving it by adding another convolution layer but that didnt result in improving the accuracy.

* Which parameters were tuned? How were they adjusted and why?

I tested by tuning Epochs, learning rate and batch size.
- On increasing the batch size the accuracy became worse so I abandoned changing it.
- I increased the Epochs to better train the model but not to increase to level where the model over fits the training data
- I decreased the learning rate to better optimize the model

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The dropout layer helps in reducing the overfitting of data specially for classes having more data.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images I have chosen are pretty similar to the ones in the dataset. The only difference seemed to be with the brightness of the images compared to the one in training data set

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     		| Right-of-way at the next intersection  									| 
| Speed limit (30km/h)    			|Speed limit (30km/h)										|
| Priority road					| Priority road											|
| Keep right      		| Keep right				 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][image11]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


