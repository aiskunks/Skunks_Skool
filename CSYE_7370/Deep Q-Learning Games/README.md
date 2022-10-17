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



