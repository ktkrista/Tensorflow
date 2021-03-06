# Tensorflow
In this project, I built two models which are an 8-percent and a 16-percent model. Our goal is to construct the models for the lowest loss value or a better accuracy rate for the validation set.

To begin with, the 8-percent model is built to be used with the 8-percent dataset. For the first model, I chose to work with the activation functions, which are the Rectified Linear Unit (“relu”) activation function for the hidden layers and Softmax activation function for the last layer. After trying different combinations of parameters, I have reached the satisfying rate of the accuracy for the dataset which is approximately 92 percent with the 24 percent loss value.


On the other hand, I have used the same approach for the 16-percent model. However, this time I decided to increase the number of dimensionality of the output space. A high number of units can introduce problems like overfitting and exploding gradient problems. On the other side, a lower number of units can cause a model to have high bias and low accuracy values. Therefore, I have tried several different random sets of parameters especially with the units argument, and decided to include some more dropouts for the best accuracy rate. Furthermore, since we have to predict the probability as an output, I chose to proceed with Sigmoid or Logistic Activation Function for the hidden layers and Softmax activation function for the last layer. The output for the 16-percent model is approximately 96 percent accuracy with a 17 percent loss value.


In conclusion, the outputs are 92 percent accuracy for an 8-percent model and 96 percent accuracy for a 16-percent model.
