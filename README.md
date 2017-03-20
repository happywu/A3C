## Still in progress.

## A3C  

This is a MXNET implementation of A3C as described in ["Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/pdf/1602.01783v1.pdf).



## Requirement

* openai gym
* mxnet


## Flappy Bird

Game source from [Using Deep Q-Network to Learn How To Play Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird). 


If you don't want to run FlappyBird, you can ignore this.

To run experiment:
```bash
python a3c.py --game-source=flappybird --num-threads=16 --save-model-prefix=a3c-flappybird --save-every=1000
```

To eval, I have upload a checkpoint of mine, you could try your own parameters. 
```bash
python a3c.py --test --model-prefix=a3ce-8 --load-epoch=305000 --game-source=flappybird
```

### Notice
If you train on computer without GPUS, please change "devs = gpu(1)" to "devs = cpu()"
