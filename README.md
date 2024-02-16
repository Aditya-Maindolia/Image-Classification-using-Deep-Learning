# Image-Classification-using-Deep-Learning
# Introduction
Machine Learning:
Machine learning (ML) is a subfield of artificial intelligence (AI) that focuses on developing 
algorithms and models that enable computers to learn from data and make predictions or 
decisions without being explicitly programmed. In traditional programming, humans provide 
explicit instructions to achieve a specific task, while in machine learning, the algorithm learns 
patterns and rules from data to perform a task.</br></br>
Deep Learning:
Deep learning is a subset of machine learning that employs artificial neural networks to model 
and solve complex tasks. These neural networks, inspired by the human brain, consist of layers 
of interconnected nodes (neurons) that process and learn representations of data. Deep learning 
has gained prominence due to its ability to automatically learn hierarchical features from raw 
data, making it particularly effective for tasks such as image and speech recognition.</br></br>
Image Classification using Deep Learning:
Image classification is a task within the realm of computer vision, where the goal is to 
categorize an image into predefined classes or labels. Deep learning, especially Convolutional 
Neural Networks (CNNs), has proven highly successful in image classification tasks. CNNs 
are designed to automatically learn hierarchical features from images, capturing patterns and 
spatial relationships.</br></br>
The typical workflow for image classification using deep learning involves the following steps:</br>
1. Data Collection: Gather a labeled dataset containing images with corresponding class 
labels.</br>
2. Data Preprocessing: Prepare the data by resizing, normalizing, and augmenting 
images to enhance the model's ability to generalize.</br>
3. Model Architecture: Design a CNN architecture suitable for the specific image 
classification task. This may involve stacking convolutional layers, pooling layers, and 
fully connected layers.</br>
4. Training: Feed the labeled training data into the model and adjust the model's 
parameters (weights and biases) using optimization algorithms to minimize the 
difference between predicted and actual labels.</br>
5. Validation: Evaluate the model's performance on a separate dataset not used during 
training to assess its generalization ability.</br>
6. Testing: Apply the trained model to unseen images to assess its real-world 
performance.</br>
Popular deep learning frameworks like TensorFlow and PyTorch provide tools and libraries to 
facilitate the implementation of image classification models

# Dataset
The dataset used in this project is taken from Ciphar-10 dataset. It comprises of 60000
images, out of which 50000 images are used for training purpose and 10000 images are for 
the purpose of testing.
# Creating Model
1. First InceptionV3 model is loaded with pre-trained weight from ImageNet into variable 
base_model.</br>
2. Then freeze all layers of base_model by setting all trainable attributes to ‘False’. This 
prevents the weights of these attributes to update thus retaining the knowledge captured 
by pre-trained model.</br>
3. Create a new sequential model.</br>
4. UpSampling2D layer is added to increase the spatial resolution of the input by a factor 
of 7 in both dimensions. This is often done to match the dimensions of the feature maps 
produced by the InceptionV3 base model.</br>
5. The pre-trained base_model is added to the new model.</br>
6. GlobalAveragePooling2D layer is added to pool the spatial information from the feature 
maps and reduce their dimensions to a fixed size.</br>
7. A dense layer with 256 units and ReLU activation is added.</br>
8. A dropout layer with a dropout rate of 0.5 is added for regularization.</br>
9. Finally, a dense layer with 10 units (assuming it's a classification task with 10 classes) 
and softmax activation is added.</br>
# Result
The model created for Image classification in the Ciphar-10 dataset gives better result than common CNN method.</br>
The common CNN method provided accuracy of 71.1%.</br>
But the new model gives 85.51% accuracy for training dataset.</br>
And 83.75% accuracy for testing dataset.</br></br>

There is an increase of 12.65% accuracy by creating model using InceptionV3

