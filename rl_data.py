import mxnet as mx
import numpy as np
import gym
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

class RLDataIter(object):
    def __init__(self, resized_width, resized_height, agent_history_length, visual=False):
        super(RLDataIter, self).__init__()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.env = self.make_env(visual)
        self.last_raw_frame = None
        self.state_buffer = deque()
        self.state_ = np.zeros((self.agent_history_length, self.resized_width, self.resized_height))
        self.provide_data = mx.io.DataDesc('data', self.state_.shape,
            np.uint8)

    def make_env(self,visual=False):
        raise NotImplementedError()

    def get_initial_state(self):
        raise NotImplementedError()

    '''
    This preprocessing method shows low performance.
    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height))
    '''

    def get_preprocessed_frame(self, frame):
        frame = frame[34:34+160, :160]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [42, 42])
        return frame

    '''
    TODO: This preprocessing method works well for atari environment, however needs to change to general one with variable 
    height and width.
    '''
    def _get_preprocessed_frame(self, img):
        img = img[35:195]
        img = img[::2, ::2, 0]
        img[img == 144] = 0
        img[img == 109] = 0
        img[img != 0] = 1
        #curr = np.expand_dims(img.astype(np.float).ravel(), axis=0)
        curr = np.expand_dims(img.astype(np.float), axis=0)
        # Subtract the last preprocessed image.
        #diff = (curr - self.prev if self.prev is not None
        #        else np.zeros((1, curr.shape[1])))
        #self.prev = curr
        return curr.reshape((80, 80))
    '''
    Follow the original Lua implementation
    '''
    '''
    def get_preprocessed_frame(self, img, crop=False):
        # Max of two consecutive frames
        img = np.maximum(img, self.last_raw_frame)
        self.last_raw_frame = img
        # RGB -> Luminance
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        img = img.astype(np.uint8)
        if crop:
             # Shrink (210, 160) -> (110, 84)
            img = cv2.resize(img, (84, 110),
                             interpolation=cv2.INTER_LINEAR)
            assert img.shape == (110, 84)
            # Crop (110, 84) -> (84, 84)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
        else:
            img = cv2.resize(img, (84, 84),
                             interpolation=cv2.INTER_LINEAR)
    '''

    def act(self, action_index):
        raise NotImplementedError()

    def data(self):
        return mx.nd.array(self.state_, dtype=np.uint8)

class GymDataIter(RLDataIter):
    def __init__(self, game, resized_width, resized_height, agent_history_length):
        self.game = game
        super(GymDataIter, self).__init__(resized_width, resized_height, agent_history_length)
        self.act_dim = self.env.action_space.n
        if (self.env.spec.id == 'Pong-v0' or self.env.spec.id == 'Breakout-v0'):
            self.act_dim = 3
        sed = np.random.randint(1000)
        self.env.seed(sed)

    def get_initial_state(self):
        # reset game, clear state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        self.last_raw_frame = x_t
        x_t = self.get_preprocessed_frame(x_t)
        # store (history information) last four frames.
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

        # initial state, agent_history_length-1 empty state
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t

    def make_env(self,visual=False):
        return gym.make(self.game)

    def act(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        if (self.env.spec.id == 'Pong-v0' or self.env.spec.id == 'Breakout-v0'):
            action_index += 1

        x_t1, r_t, terminal, info = self.env.step(action_index)

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

class FlappyBirdIter(RLDataIter):
    def __init__(self, resized_width, resized_height, agent_history_length, visual=False):
        import sys
        sys.path.append('game/')
        import wrapped_flappy_bird as game
        self.act_dim = 2
        super(FlappyBirdIter, self).__init__(resized_width, resized_height, agent_history_length, visual)

    def make_env(self, visual=False):
        game = FlappyBird()
        return PLE(game, fps=30, display_screen=visual)

    def get_initial_state(self):

        # reset game, clear state buffer
        self.state_buffer = deque()

        self.env.init()
        x_t = self.env.getScreenRGB()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

        # initial state, agent_history_length-1 empty state
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t

    def act(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        action_set = self.env.getActionSet()
        r_t = self.env.act(action_set[action_index[0]])
        x_t1 = self.env.getScreenRGB()
        x_t1 = self.get_preprocessed_frame(x_t1)
        #print sum((x_t1))
        terminal = self.env.game_over()
        info = None

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        self.state_ = s_t1

        return s_t1, r_t, terminal, info

class MultiThreadFlappyBirdIter(RLDataIter):
    def __init__(self, resized_width, resized_height, agent_history_length, visual=False):
        self.act_dim = 2
        super(MultiThreadFlappyBirdIter, self).__init__(resized_width, resized_height, agent_history_length, visual)
        self.get_initial_state()

    def make_env(self, visual=False):
        return game.GameState()

    def get_initial_state(self):

        # reset game, clear state buffer
        self.state_buffer = deque()
        action0 = np.array([1, 0])
        image, _, _ = self.env.frame_step(action0)
        x_t = image
        x_t = (self.preprocess(x_t))
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

        # initial state, agent_history_length-1 empty state
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t
    def preprocess(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (self.resized_width, self.resized_height)), cv2.COLOR_BGR2GRAY)
        #ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,(self.resized_width, self.resized_height))

    def act(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        action = np.zeros(2)
        action[action_index] = 1
        x_t1, r_t, terminal = self.env.frame_step(action)
        x_t1 = (self.preprocess(x_t1))
        #print sum((x_t1))
        info = None

        p_s =  self.state_buffer
        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        #print (x_t1), len(p_s)
        #print 'f', p_s
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        self.state_ = s_t1
        #print 'j', self.state_buffer
        #print 'quea', ((p_s == self.state_buffer))
        #print 'state~~', s_t1

        return s_t1, r_t, terminal, info
