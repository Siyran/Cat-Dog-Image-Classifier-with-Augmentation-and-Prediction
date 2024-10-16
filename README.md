This project builds a binary image classifier using TensorFlow and Keras to distinguish between cats and dogs. It includes the complete pipeline from data preprocessing and validation to model training and image prediction.

The code validates the dataset by checking for and removing corrupted images to ensure data quality. It also applies data augmentation during training, using techniques such as rotation, zooming, and flipping to improve the model’s ability to generalize.

The classifier is a Convolutional Neural Network (CNN) with multiple convolutional, max-pooling, and dense layers. The model is trained using the `RMSprop` optimizer and `binary_crossentropy` loss function, designed specifically for binary classification tasks. Training is done with real-time augmentation, and the validation dataset helps monitor performance across multiple epochs.

A visualization function plots the training and validation accuracy and loss metrics over time, providing insights into the model’s learning process.

After training, the model is saved as an H5 file for future use. The project also includes a prediction function that allows users to input a new image and receive a classification (cat or dog). The model resizes and preprocesses the input image before predicting the class based on the trained network.

This project offers a streamlined solution for creating, training, and deploying a binary image classifier for cats and dogs.



Data Set Link: https://www.microsoft.com/en-us/download/details.aspx?id=54765
