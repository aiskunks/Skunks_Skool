
# Dogs and Cats Binary Images Classification With Convolutional Network

Zeyuan Zhu<br>
Haoyuan Qin

Python 3.8<br>
tensorflow_macos-2.11.0

Deep learning has become one of the most important breakthroughs in artificial intelligence over the past decade. It contains a variety of methods, including neural networks, hierarchical probabilistic models, and many specific unsupervised and supervised feature-learning algorithms. 
In this project, we built a module that could recognize the image of dogs & cats by using Keras as the deep learning framework. The dataset from Kaggle contains 25000 pictures (12500 cats & 12500 dogs) training sets are created based on these pictures. Firstly, training and validate the network by original pictures 30 epochs. The validation accuracy is 75%, and the training accuracy is close to 100%. The reason for this is the module was overfitting these data. Then we modify train pictures by using data augmentation which changed these pictures randomly to make sure that our module will not receive duplicated pictures. And the validation accuracy after data augmentation is up to 80% and training accuracy is also closing to 80% after 100 epochs.

![image](https://user-images.githubusercontent.com/71553583/208253814-03cda16e-832b-41ab-b871-a671bb0ae93c.png)


### Data Set
https://www.kaggle.com/competitions/dogs-vs-cats/overview

![image](https://user-images.githubusercontent.com/71553583/208253741-a391d4e0-8550-4465-8456-6c00a09eb81a.png)
![image](https://user-images.githubusercontent.com/71553583/208253745-4ed2de6c-da9e-479e-b6cb-74ee8b59c076.png)
