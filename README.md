# Hackathon Project - Deep Learning for Dementia
In the context of TechCrew's Hackathon'23, I participated and achieved the 2nd place with an Dementia detection model. The competition revolved around machine learning, and my project aimed to create a reliable system for identifying Dementia utilizing deep learning methodologies. Despite the limited time frame and modest dataset, the model's performance was notably strong in classification.

![image](https://github.com/oguzgnrn/HackathonProject-DeepLearningForDementia/assets/96068121/302d3bba-b788-44e0-8a59-b82c5bfa87f7)

## Dataset's Complexity
The crux of my project revolved around a dataset sourced from Kaggle, consisting of 6400 MRI images categorized into four distinct classes, each illuminating a different facet of Dementia progression:
Mild Demented (896 images): Illustrating individuals in the early stages of mild dementia symptoms.
Moderate Demented (64 images): Capturing images of individuals exhibiting moderate dementia symptoms.
Non Demented (3200 images): Portraying individuals devoid of any dementia symptoms.
Very Mild Demented (2240 images): Highlighting early-stage dementia manifestations in individuals.


![image](https://github.com/oguzgnrn/HackathonProject-DeepLearningForDementia/assets/96068121/3a7b3ba2-a914-4663-bd7a-5216ce21037c)


## Engineering a Model of Precision

During the early stages of the project, I encountered the common challenge of achieving satisfactory accuracy levels in my Dementia detection model. Initially, the model's accuracy hovered between 70% and 80%, which prompted a need for a more refined approach. With determination, I iterated through various model architectures and techniques, ultimately arriving at a configuration that significantly elevated the accuracy to an impressive 91%.
The pivotal transformation occurred when I introduced a new model architecture, which allowed for more complex and nuanced feature extraction from the dataset. The revised model architecture is as follows:

```
from keras import layers

model = tf.keras.Sequential([
 layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Dropout(0.5),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(4, activation="softmax")
])

model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"])

epochs = 5
history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data,
                    batch_size=batch_size)
```

![image](https://github.com/oguzgnrn/HackathonProject-DeepLearningForDementia/assets/96068121/5c1a5cf5-9d0e-4d45-a667-a2ca9a8ab9b8)


_In this code block, we construct a sequential neural model using TensorFlow and Keras. This model architecture comprises essential layers like convolutional and pooling layers for feature extraction from images, dropout layers to enhance generalization, and dense layers for classification. With the model compiled using the Adam optimizer and a loss function, we embark on a training journey spanning five epochs. Throughout this process, the model learns from the training data while refining its performance through validation on a separate dataset_

The introduction of this architecture was transformative. The model began to demonstrate a remarkable ability to discern intricate patterns within the dataset, which significantly boosted its accuracy. The utilization of this architecture in conjunction with the appropriate choice of hyperparameters and optimization techniques paved the way for the model's final accuracy of 91%.


## Creating a User-Friendly Interface with Gradio

To ensure a user-friendly experience, I incorporated the Gradio library, which allowed me to develop an interface that makes interactions straightforward. This interface enables users to easily upload MRI images and receive immediate predictions about the likelihood of dementia. By combining advanced technology with simplicity, this user-centric design embodies the project's aim of harnessing data-driven solutions for practical results.
  
  ```
  import tensorflow as tf
  import matplotlib.pyplot as plt
  import gradio as gr
  import numpy as np
  from tensorflow.keras.preprocessing.image import load_img, img_to_array
  
  loaded_model = tf.keras.models.load_model("hackathon/modelDusuk")
  
  def classify_image(img):
      image_array = img_to_array(img)
      image_array = np.expand_dims(image_array, axis=0)
      predictions = loaded_model.predict(image_array)
      class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented',
                     'Very_Mild_Demented']
      predicted_class_index = np.argmax(predictions[0])
      predicted_class = class_names[predicted_class_index]
      confidence = predictions[0][predicted_class_index]
      confidence = confidence*100
      ret = f"{confidence:.2f}%   {predicted_class}"
      return ret
  
  image = gr.inputs.Image(shape=(224,224))
  label = gr.outputs.Label(num_top_classes=1)
  
  gr.Interface(fn=classify_image,
               inputs=image,
               outputs=label,
               examples=["hackathon/mild_580.jpg",
                         "hackathon/mild_815.jpg",
                         "hackathon/moderate_9.jpg",
                         "hackathon/moderate_8.jpg",
                         "hackathon/non_506.jpg",
                         "hackathon/non_471.jpg",
                         "hackathon/verymild_9.jpg",
                         "hackathon/verymild_255.jpg"]).launch(share=True,debug=True)
  ```
_1-Loading the Model: The TensorFlow library is employed to load a pre-trained AI model that has been trained to recognize patterns in MRI images related to dementia._

_2-Classification Function: The function classify_image takes an input image, processes it using TensorFlow and NumPy, and passes it through the loaded model. This results in predictions for different classes of dementia: 'Mild Demented', 'Moderate Demented', 'Non Demented', and 'Very Mild Demented'. The function computes the confidence level of the prediction and formats the result._
 
_3-User-Friendly Interface: The Gradio library is utilized to create a user-friendly interface. Users can upload MRI images of size 224x224 pixels, and the model will provide predictions based on these images._
 
_4-Real-Time Interaction: Users can directly interact with the interface by uploading MRI images. The model processes the image through the classify_image function and displays the predicted class of dementia along with its confidence percentage._
 
_5-Launch and Sharing: By invoking .launch(share=True, debug=True), the interface is launched, allowing users to experience real-time predictions and share the tool with others. The debug mode provides insights for troubleshooting if needed._

## Conclusion
Securing the 2nd place in Hackathon'23 magnifies the prowess of strategic model architecture and innovation within the realm of machine learning. It underscores the potential of machine learning techniques in healthcare despite dataset limitations. This journey epitomizes the iterative essence of machine learning, exemplified by the model's evolution from modest accuracy to a stalwart dementia detection tool. This accomplishment fuels my drive to extend the model's impact, contributing significantly to dementia detection efforts. Ultimately, this venture illuminates the transformative potency of persistence, innovation, and data-driven approaches in reshaping the frontiers of technology and healthcare.

![image](https://github.com/oguzgnrn/HackathonProject-DeepLearningForDementia/assets/96068121/ecefb9f8-a31d-4182-b3bb-6d26da3c567b)

You can find the dataset from here: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset





















