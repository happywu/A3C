import mxnet as mx
import numpy as np
import gym
import cv2
import math
import Queue
from threading import Thread
import time
from flask import Flask, render_template, Response

def env_step(args):
    return args[0].step(args[1])

class RLDataIter(object):
    def __init__(self, input_length, web_viz=True):
        super(RLDataIter, self).__init__()
        self.env = self.make_env()
        self.state_ = None
        self.input_length = input_length
        self.act_dim = self.env[0].action_space.n


        self.reset()

        self.provide_data = [mx.io.DataDesc('data', self.state_.shape,
            np.uint8)]

        self.web_viz = web_viz

        if web_viz:
            self.queue = Queue.Queue()
            self.thread = Thread(target=make_web, args=(self.queue,))
            self.thread.daemon = True
            self.thread.start()

    def make_env(self):
        raise NotImplementedError()

    def reset(self):
        #self.state_ = np.tile(
        #    np.asarray(self.env.reset(), dtype=np.uint8).transpose((2, 0, 1)),
        #    (1, self.input_length, 1, 1))
        self.state_ = np.tile(
                np.asarray([env.reset() for env in self.env],
                    dtype=np.uint8).transpose((0, 3, 1, 2)), 
                (1, self.input_length, 1, 1))

    def visual(self):
        raise NotImplementedError()

    def act(self, action):
        new = [env.step(act) for env, act in zip(self.env, action)]
        #new = [self.env.step(action)]

        reward = np.asarray([i[1] for i in new], dtype=np.float32)
        done = np.asarray([i[2] for i in new], dtype=np.float32)


        channels = self.state_.shape[1]/self.input_length
        state = np.zeros_like(self.state_)
        state[:,:-channels,:,:] = self.state_[:,channels:,:,:]
        for i, (ob, env) in enumerate(zip(new, self.env)):
            if ob[2]:
                state[i,-channels:,:,:] = env.reset().transpose((2,0,1))
            else:
                state[i,-channels:,:,:] = ob[0].transpose((2,0,1))
        self.state_ = state

        if self.web_viz:
            try:
                while self.queue.qsize() > 10:
                    self.queue.get(False)
            except Empty:
                pass
            frame = self.visual()
            self.queue.put(frame)

        return reward, done

    def data(self):
        return [mx.nd.array(self.state_, dtype=np.uint8)]


class GymDataIter(RLDataIter):
    def __init__(self, game, input_length, web_viz=False):
        self.game = game
        super(GymDataIter, self).__init__(input_length, web_viz=web_viz)

    def make_env(self):
        return [gym.make(self.game)]

    def visual(self):
        data = self.state_[:4, -self.state_.shape[1]/self.input_length:, :, :]
        return visual(np.asarray(data, dtype=np.uint8), False)
