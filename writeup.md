# **Behavioral Cloning** 

## Writeup

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model comes from the NVIDIA Model mentioned in the project lesson.  I also implemented a lenet architecture, but it does not perform as well as the NVIDIA architecture.  

#### 2. Model parameter tuning

I played around with various parameters the first attempt I had with this project, but when I restarted it, I used a YAGNI approach and found that I didn't really need to modify any parameters.  I started out my second attempt using no special parameters except for Epochs, which I started out with 5.  I found no reason to change this.  I also started out with the adam optimer and the MSE loss function, both of which I had no reason to change.

#### 3.  Training data
At First, I tried creating my own data by driving around the track 10 times, and then 10 times in reverse.  This data set did not train well at all, and eventually I scrapped this data set for Udacity's own data set.  I found that I could not collect data as good as udacity's, so this is the only data set I used. 

I did augment the udacity data by using the left and right camera images, as well as flipping the images.  This led me to have 6 images and measurements per line in the CSV file.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I greatly struggled with this project the first time I attempted it, so much so that I threw away my first attempt and started over from scratch.  This led to a YAGNI approach that started out by using the lenet architecture, and then eventually implementing the NVIDIA model once it seemed that I had reached the maximum performance I was going to get with Lenet.  I came to this conclusion because the LeNet Model kept missing the turn after the bridge, and once that happened no matter how much I played around with the model, I switched to the NVIDIA model.

As far as data, I used the Udacity Provided Data. I used the Left, Right and Center Images, and then flipped them and used those as well.  I used a correction factor of 0.2 for the Left and Right images.  Something else I did was throw away 90% of the images that had a steering angle of 0.  Since the majority of the track can be solved by the car driving straight, and the turns being the tricky parts, this helped my model by providing a greater percentage of turning data.  This also helped my model train faster. I split my data into training and validation data with an 80/20 split. 

I also did not use a generator, as my AWS instance was able to handle the data without any out of memory errors. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
