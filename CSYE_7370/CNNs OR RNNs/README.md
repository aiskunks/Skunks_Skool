# Deep Learning with CNNs OR RNNs (Sequence Models)

You have a choice of playing with and learning CNNs or RNNs (Sequence Models). While choosing whatever you may already know may be easier I suggest that you play with (i.e. learn) whichever you are least familiar.

## Choice A- Deep Learning with CNNs

Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the notebooks can be run.


Classify images with CNNs 


### PART A - DEEP LEARNING MODEL (40 POINTS)

Find an image dataset. It cannot be MNIST or CFFAR but can be TMNIST. For TMNIST-Alphabet (94 characters and over 281,000 images) see https://www.kaggle.com/nikbearbrown/tmnist-alphabet-94-characters.


Can be done with PyTorch, TensorFlow, and/or Keras


### PART B - ACTIVATION FUNCTION (10 POINTS)


On your Deep Learning model data
Change the activation function. How does it affect accuracy? How does it affect how quickly the network plateaus?


**Various activation functions:** 

Rectified linear unit (ReLU)<br>
TanH<br>
Leaky rectified linear unit (Leaky ReLU)<br>
Parameteric rectified linear unit (PReLU) Randomized leaky rectified linear unit (RReLU)<br>
Exponential linear unit (ELU)<br>
Scaled exponential linear unit (SELU)<br>
S-shaped rectified linear activation unit (SReLU)<br>
Adaptive piecewise linear (APL)<br>

### PART C - COST FUNCTION (10 POINTS)  On your Deep Learning model.  

Change the cost function. How does it affect accuracy? How does it affect how quickly the network plateaus?


**Various forms of cost:**
Quadratic cost (mean-square error)<br>
Cross-Entropy<br>
Hinge<br>
Kullback–Leibler divergence<br>
Cosine Proximity<br>
User-defined<br>
And many more, see https://keras.io/losses/<br>

### PART D - EPOCHS (10 POINTS)

On your Deep Learning model data<br>
Change the number of epochs initialization. How does it affect accuracy?<br>
How quickly does the network plateau?<br>

### PART E - GRADIENT ESTIMATION (10 POINTS)
On your Deep Learning model

Change the gradient estimation. How does it affect accuracy? How does it affect how quickly the network plateaus?<br>
**Various forms of gradient estimation:**
Stochastic Gradient Descent<br>
Adagrad<br>
RMSProp<br>
ADAMN<br>
AGAdadelta<br>
Momentum<br>

### PART F - NETWORK ARCHITECTURE (10 POINTS)

On your Deep Learning model change the network architecture. How does it affect accuracy? How does it affect how quickly the network plateaus?
**Various forms of network architecture:**
Number of layers<br>
Size of each layer<br>
Connection type<br>

### PART G - NETWORK INITIALIZATION (10 POINTS)
On your Deep Learning model


Change the network initialization. How does it affect accuracy? How does it affect how quickly the network plateaus?


**Various forms of network initialization:**
0<br>
UniformGaussian<br>
Xavier<br>
Glorot Initialization http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initializationXavier<br>
Uniform<br>
Xavier Gaussian<br>


**YouTube**<br>
PyTorch & TensorBoard<br>
https://www.youtube.com/watch?v=pSexXMdruFM.<br>
 

Hyperparameter Tuning and Experimenting PyTorch & TensorBoard<br>
https://www.youtube.com/watch?v=ycxulUVoNbk.

## Choice B- Deep Learning with RNNs

Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the notebooks can be run.


Note that Hugging Face has an excellent python library and downloadable language models<br> 
https://huggingface.co/<br>
Often these models can be run in one or two lines of python code so the grading will emphasize the explanation of what the models are doing and the interpretation of the results.<br>


1. Fill-Mask (10 Points)<br>
Run a <Fill-Mask> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

2. Question Answering (10 Points)<br>
Run a <Question Answering> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

3. Summarization (10 Points)<br>
Run a <Summarization> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

4. Text Classification (10 Points)<br>
Run a <Text Classification> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

5. Text Generation (10 Points)<br>
Run a <Text Generation> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

6. Text2Text Generation (10 Points)<br>
Run a <Text2Text> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

7. Token Classification (10 Points)<br>
Run a <Token Classification> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

8. Translation (10 Points)<br>
Run a <Translation> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

9. Zero-Shot Classification (10 Points)<br>
Run a <Zero-Shot> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.

10. Sentence Similarity (10 Points)<br>
Run a <Sentence Similarity> language model. Explain the theory behind your model, and run it.  Analyze how well you think it worked.


https://www.youtube.com/watch?v=ycxulUVoNbk