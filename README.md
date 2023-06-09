# Facial-Expression-Recogination✨
![3-Figure1-1](https://github.com/codejay12/Facial-Expression-Recogination/assets/97224357/bbadf9c6-8449-4f18-be11-08b5ea00ede3)

 
Facial expression recognition using PyTorch is a powerful computer vision technique that allows machines to identify and understand the emotions conveyed by human faces. PyTorch, a popular deep learning framework, provides a flexible and efficient platform for building and training facial expression recognition models.




**The process of facial expression recognition using PyTorch typically involves the following steps:**

# Dataset Preparation:
- A diverse dataset of facial images with corresponding emotion categories such as happy, sad, angry, surprised, etc. These images can be obtained kaggle.
- ` Dataset link `: [ link ](https://www.kaggle.com/datasets/ashishpatel26/fer2018)


# Data Preprocessing: 
- The facial images are preprocessed to enhance their quality and standardize their format.
- Common preprocessing techniques include resizing the images, normalizing pixel values, and applying image augmentation methods to increase the diversity of the dataset.


# Model Architecture Design: 
- PyTorch offers a wide range of neural network modules and tools to design the architecture of the facial expression recognition model.
- Typically, convolutional neural networks (CNNs) are used for this task due to their ability to capture spatial relationships in images effectively. The architecture may include multiple convolutional layers, pooling layers, and fully connected layers.
- Parameters used
```
epochs = 15
lr = 0.0005
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=5e-2, lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
```

# 1. Base model 17 
- `The model architecture consists of a total of 17 layers. It includes 14 convolutional layers, 2 fully connected layers, and 1 flattening layer. These layers are designed to extract features from the input images and make predictions for image classification tasks. It includes a total of four convolutional layers, followed by two fully connected layers. Additionally, there are ReLU activation functions, batch normalization layers, max pooling layers, and dropout layers employed throughout the network for improved performance in image classification tasks.
`![base model results](https://github.com/codejay12/Facial-Expression-Recogination/assets/97224357/0dc069e3-1e4c-484c-8fbd-e94f8e764d74)


# 2.Resnet Model
- `Resnet architecture -ResNet (Residual Neural Network) is a deep convolutional neural network architecture that has shown remarkable performance in various computer vision tasks, including facial expression recognition. ResNet addresses the challenge of training very deep neural networks by introducing residual connections, or skip connections, that enable the network to learn residual mappings instead of direct mappings.`

![Resnet results](https://github.com/codejay12/Facial-Expression-Recogination/assets/97224357/14b6f80d-dd6c-4083-9ac8-e3c09df788a6)


# Model Training: 
- The prepared dataset is divided into training and validation sets. 
- The model is trained using the training set, where the weights of the network are updated iteratively to minimize the loss between predicted and ground truth emotions.
- The loss function used can be categorical cross-entropy, which measures the dissimilarity between predicted and actual emotion labels.


# Model Evaluation: 
- The trained model is evaluated using the validation set to assess its performance and fine-tune hyperparameters. 
- Metrics such as accuracy, precision, recall, and F1 score can be used to measure the model's effectiveness in recognizing facial expressions.


# Testing and Deployment:
- Once the model achieves satisfactory performance, it can be tested on new, unseen facial images to predict the emotions expressed. 
- The model can be integrated into real-world applications, such as emotion-aware systems, human-computer interaction interfaces, or facial expression analysis in research studies.

# Model monitoring:
- `Evidently AI` is a crucial process that involves continuously assessing the performance and behavior of machine learning models deployed in production. 
- Evidently AI provides a comprehensive platform for model monitoring, offering a range of tools and metrics to help monitor and analyze model performance, data quality, and model drift over time.
- Base model 17 layers

![results_base model](https://github.com/codejay12/Facial-Expression-Recogination/assets/97224357/818ca3e0-adb9-46fa-b600-7610c39184b4)

- Resnet model 

![Resnet model output](https://github.com/codejay12/Facial-Expression-Recogination/assets/97224357/1029d426-ad53-410e-bdc0-f17aa7c8b9d5)

# Finally model with Base model with 17 layers take into consdieration for model production 






