### RNN_Poetry_Generator

This is a chinese poem generator based on recurrent neural network, which mainly uses Tang poetry data set to train the model. 
Using the trained model, randomly generate a chinese poem, write a continuation of an chinese poem, and randomly generate a Tibetan poem.

Randomly generate a chinese poem:

```
水泉临鸟声，北去暮空行。
林阁多开雪，楼庭起洞城。
夜来疏竹外，柳鸟暗苔清。
寂寂重阳里，悠悠一钓矶。
```

Write a continuation of a poem ("床前明月光" as an example):

```
床前明月光，清水入寒云。
远景千山雨，萧花入翠微。
影云虚雪润，花影落云斜。
独去江飞夜，谁能作一花。
```

Randomly generate a acrostic poem ("海阔天空" as an example):

```
海口多无定，
阔庭何所难。
天山秋色上，
空石昼尘连。
```

### Start

First train the model: run the command`python3 train.py`.

Then use trained models to generate poems: run the command`python3 eval.py`.