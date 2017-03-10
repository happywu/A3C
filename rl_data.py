import mxnet as mx
import numpy as np
import gym
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import cv2
import math
import Queue
from threading import Thread
import time
from flask import Flask, render_template, Response

def env_step(args):
    return args[0].step(args[1])

class RLDataIter(object):
    def __init__(self, resized_width, resized_height, agent_history_length):
        super(RLDataIter, self).__init__()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.env = self.make_env()

        self.act_dim = self.env.action_space.n
        if (self.env.spec.id == 'Pong-v0' or self.env.spec.id == 'Breakout-v0'):
            self.act_dim = 3

        self.state_buffer = deque()
        self.state_ = np.zeros((self.agent_history_length, self.resized_width, self.resized_height))
        self.provide_data = [mx.io.DataDesc('data', self.state_.shape,
            np.uint8)]

    def make_env(self):
        raise NotImplementedError()

    def get_initial_state(self):

        # reset game, clear state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

        # initial state, agent_history_length-1 empty state
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height))

    def act(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        self.state_ = s_t1

        return s_t1, r_t, terminal, info

    def data(self):
        return mx.nd.array(self.state_, dtype=np.uint8)

class GymDataIter(RLDataIter):
    def __init__(self, game, resized_width, resized_height, agent_history_length):
        self.game = game
        super(GymDataIter, self).__init__(resized_width, resized_height, agent_history_length)

    def make_env(self):
        return gym.make(self.game)
