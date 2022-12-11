### Let's predict which team will fetch you the most points 
What is Multi-Armed Bandit?
The multi-armed bandit problem is a classic problem that well demonstrates the exploration vs exploitation dilemma. Imagine you are in a casino facing multiple slot machines and each is configured with an unknown probability of how likely you can get a reward at one play. The question is: What is the best strategy to achieve highest long-term rewards?<br> <br>

In this post, we will only discuss the setting of having an infinite number of trials. The restriction on a finite number of trials introduces a new type of exploration problem. For instance, if the number of trials is smaller than the number of slot machines, we cannot even try every machine to estimate the reward probability (!) and hence we have to behave smartly w.r.t. a limited set of knowledge and resources (i.e. time).

![image](https://user-images.githubusercontent.com/91396776/192071624-5b884b1d-e672-48c3-aa58-7e4cac4ec6e2.png)

A naive approach can be that you continue to playing with one machine for many many rounds so as to eventually estimate the “true” reward probability according to the law of large numbers. However, this is quite wasteful and surely does not guarantee the best long-term reward.

Bandit Strategies
Based on how we do exploration, there several ways to solve the multi-armed bandit.

No exploration: the most naive approach and a bad one.
Exploration at random
Exploration smartly with preference to uncertainty
