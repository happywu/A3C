## Still in progress.

## A3C  

This is a MXNET implementation of A3C as described in ["Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/pdf/1602.01783v1.pdf).



## Requirement

* openai gym
* mxnet


## Flappy Bird

Game source from [Using Deep Q-Network to Learn How To Play Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird). 

Running FlappyBird needs [PyGame-Learning-Environment(PLE)](https://github.com/ntasfi/PyGame-Learning-Environment). 

If you don't want to run FlappyBird, you can ignore this.

To run experiment:
```bash
python async_dqn.py --game-source='flappybird'
```

### Installation

PLE requires the following dependencies:
* numpy
* pygame
* pillow

Clone the repo and install with pip.

```bash
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
``` 