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
 

Hyperparameter Tuning and Experimenting PyTorch & TensorBoard


https://www.youtube.com/watch?v=ycxulUVoNbk.

## Choice B- Deep Learning with RNNs

Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the notebooks can be run.


Note that Hugging Face has an excellent python library and downloadable language models   https://huggingface.co/.
Often these models can be run in one or two lines of python code so the grading will emphasize the explanation of what the models are doing and the interpretation of the results.
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

.











# Deep Q-Learning with an Atari-like game (Or any non-toy* Open AI gym environment)

Note: I'm changing the assignment to allow only DQL with an Atari-like game as for some reason there are few tutorials on non-DQL on an Atari-like game. Creating a tutorial using non-DQL on an Atari-like game can be a mini-project. You will still have to explain the SARSA algorithm but not implement it.


*Non-toy means any but toy text or classic control is fine [https://gym.openai.com/envs/](https://gym.openai.com/envs/)


In this assignment, you will apply Deep Q Learning to a game like any of those in the Open AI Gym Atari environments or one that you write from scratch.


[https://openai.com/blog/gym-retro/](https://openai.com/blog/gym-retro/)<br>
[https://gym.openai.com/envs/#atari](https://gym.openai.com/envs/#atari)


You can create your own game or find a similar Open AI Gym  environments


**Useful Tutorial:**<br>
Deep Reinforcement Learning for Atari Games Python Tutorial | AI Plays S...


[https://youtu.be/hCeJeq8U0lo](https://youtu.be/hCeJeq8U0lo)

## Instructions

1. Establish a baseline performance. How well did your Deep Q-learning do on your problem? (5 Points)<br>
    For example

    total_episodes = 5000<br>
    total_test_episodes = 100<br>
    max_steps = 99<br>
    learning_rate = 0.7<br>
    gamma = 0.8<br>
    epsilon = 1.0<br>
    max_epsilon = 1.0<br>
    min_epsilon = 0.01<br>
    decay_rate = 0.01 With this baseline performance, our RL program with the Taxi-v2 Toy text gives us a score of 8.13 which is considerably not bad.<br>


2. What are the states, the actions, and the size of the Q-table? (5 Points)
  

3. What are the rewards? Why did you choose them? (5 Points)
 

4. How did you choose alpha and gamma in the Bellman equation? Try at least one additional value for alpha and gamma. How did it change the baseline performance?  (5 Points)
 

5. Try a policy other than e-greedy. How did it change the baseline performance? (5 Points)
 

6. How did you choose your decay rate and starting epsilon? Try at least one additional value for epsilon and the decay rate. How did it change the baseline performance? What is the value of epsilon when if you reach the max steps per episode? (5 Points)
 

7. What is the average number of steps taken per episode? (5 Points)


8. Does Q-learning use value-based or policy-based iteration? (5 Points)<br>
Explain, not a yes or no question. 

9. Could you use SARSA for this problem? (5 Points)<br>
Explain, not a yes or no question. 


10. What is meant by the expected lifetime value in the Bellman equation? (5 Points)<br>
Explain, not a yes or no question. 

 
11. When would SARSA likely do better than Q-learning? (5 Points)<br>
Explain, not a yes or no question. 

 
12. How does SARSA differ from Q-learning? (5 Points)<br> 
Details including pseudocode and math.

 
13. Explain the Q-learning algorithm. (5 Points)<br>
Details including pseudocode and math. 

 
14. Explain the SARSA algorithm. (5 Points)<br>
Details including pseudocode and math. 

 
15. What code is yours and what have you adapted? (5 Points)<br>
You must explain what code you wrote and what you have done that is different. Failure to cite ANY code will result in a zero for this section.

 
16. Did I explain my code clearly? (10 Points)<br>
Your code review score will be scaled to a range of 0 to 10 and be used for this score.

 
17. Did I explain my licensing clearly? (5 Points)<br>
Failure to cite a clear license will result in a zero for this section.

 
18. Professionalism (10 Points)<br>
Variable naming, style guide, conduct, behavior, and attitude.



