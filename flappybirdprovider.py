# Step 1: init BrainDQN
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np

def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def dataPrep(data):
    if data.shape[2]>1:
        mean = np.array([128, 128, 128,128])
        reshaped_mean = mean.reshape(1, 1, 4)
    else:
        mean=np.array([128])
        reshaped_mean = mean.reshape(1, 1, 1)
    img = np.array(data, dtype=np.float32)
    data = data - reshaped_mean
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    data = np.expand_dims(data, axis=0)
    return data
# Step 3.2: run the game

class GameDataIter():
    def __init__(self, visualize=False):
        #super(GameDataIter, self).__init__()
        self.game = game.GameState()
        action0 = np.array([1, 0])
        image, reward, terminal = self.game.frame_step(action0)
        self.state_ = dataPrep(preprocess(image))
        self.reward = reward
        self.terminal = terminal
    def act(self, action):
        action0 = np.zeros(2)
        action0[action] = 1
        imagedata, reward, terminal = self.game.frame_step(action0)
        self.state_ = dataPrep(preprocess(imagedata))
        return [imagedata, reward, terminal]
    def state(self):
        return self.state_


