# Drowsiness-Detection using ResNet-50

# Abstract:
One of the main contributing factors to traffic accidents is drowsiness. According 
to studies, microsleeps put us at a higher risk behind the wheel than texting, being 
distracted, or being intoxicated. Thus, an automobile system that includes 
drowsiness detection can help determine whether the driver is asleep or not and 
alert the driver to avoid any such accidents. Here we focus on detecting drowsiness 
using neural network-based methodologies by monitoring eyes and other facial 
features along with the movements of the head. 
We detect the eye state of the driver and analyze if the driver is sleeping or not 
from the MRL dataset with IR images. These images are made to go through 
CNN(Convolution Neural Network) to check whether the driver’s condition is 
matching the drowsy condition. In this project, we are using Resnet-50(Residual 
Neural Network) which is a CNN that is 50 layers deep.

# Dataset:
Our dataset is a part of the MRL dataset containing infrared images of low and high 
resolution under different conditions. It is prepared for classification tasks and 
comes with various annotations like subject ID, Image ID, glasses, eye state - open 
or closed, reflections, lighting and sensor ID.
Link: https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection

# Overall Design

![image](https://github.com/Srividya083/Drowsiness-Detection/assets/145384296/248f9774-23a8-4697-bb0b-84dbd33925c4)

# Methodology:
The methodology followed for driver drowsiness detection using ResNet50 
includes several steps:

1. Data Collection: A dataset of images with both open and closed eyes of 
drivers is collected. Our dataset is uploaded to google drive in .zip format 
which is later mounted to google colab and unzipped to mrl_dataset 
folder.

2. Data Preprocessing: The collected images are preprocessed using resizing 
and data augmentation techniques. Resizing is done to convert all the 
images to the same size, and data augmentation is done to generate more 
training data and reduce overfitting.

● Resizing and Data Augmentation: The pre-processing techniques 
used in this methodology include resizing and data augmentation. 
Resizing is done to standardize the size of all images in the dataset (we 
use the standard size to be 224*224), which is necessary for feeding 
the data into the ResNet50 model. Data augmentation techniques 
such as rotation, horizontal and vertical flipping, and zooming are used 
to generate more training data and prevent overfitting. These 
techniques create variations of the original dataset, which makes the 
model more robust and helps it generalize better to new images.

3. Model Selection: ResNet50 model is selected for driver drowsiness 
detection. This model is a deep neural network architecture that is pre-trained on ImageNet dataset.
● ResNet50: The ResNet50 model is a deep neural network architecture 
that is pre-trained on ImageNet dataset. It consists of several layers 
that can detect various features in an image. The last layer of the 
model is a categorical classifier that predicts if the driver is drowsy or 
not.

4. Model Training: The ResNet50 model is trained on the preprocessed 
dataset. The training is done using the Keras library in Python. The training 
is done using an Adam optimizer with categorical cross-entropy loss.
● Adam Optimizer: During model training, the optimizer used is Adam, 
which is an adaptive learning rate optimization algorithm that is well 
suited for deep learning. 
● Categorical cross-entropy loss is used to measure the difference 
between the predicted output and the actual output.

Adam optimizer: w = w - learning_rate * m / (sqrt(v) + epsilon) 
where w is the weight parameter, m is the 1st moment vector, v is the 
2nd moment vector, learning_rate is the learning rate, and epsilon is 
a small value to prevent division by zero.

Categorical cross-entropy: loss = - sum(y_true * log(y_pred)) 
where y_true is the one-hot encoded true labels, y_pred is the 
predicted probabilities.

Softmax activation: softmax(x_i) = exp(x_i) / sum(exp(x))
where x_i is the i-th element of the input vector.

Dense layer : y = activation(dot(x, W) + b)
where x is the input tensor, W is the weight matrix, b is the bias vector, 
activation is the activation function.

5. Model Evaluation: After the model is trained, it is evaluated using a 
validation dataset. The evaluation is done using accuracy and loss metrics.

7. Model Deployment: After the model is evaluated and found to be 
satisfactory, it can be deployed for driver drowsiness detection. The 
model can be used to predict if the driver is drowsy or not by inputting a 
new image of the driver.

Once the model is trained and evaluated, it can be deployed for driver drowsiness 
detection by inputting a new image of the driver. The model predicts if the driver 
is drowsy or not based on the learned patterns in the dataset.
