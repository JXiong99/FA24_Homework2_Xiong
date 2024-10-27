import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 2500


# this is REINFORCE Agent for GridWorld
class ReinforceAgent:
    def __init__(self):
        self.load_model = True
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        # Placeholder tensor for actions chosen during an episode, with shape [batch_size, action_size]
        action = K.placeholder(shape=[None, 5])
        # Placeholder tensor for discounted rewards associated with each action, with shape [batch_size,]
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
         # Computes the probability of the selected actions by summing over the output probabilities
         # based on actions taken (1 for chosen action, 0 for others) in 'action'
        action_prob = K.sum(action * self.model.output, axis=1)
         # Calculates the policy's cross-entropy loss weighted by the discounted rewards
         # This will encourage actions that lead to higher rewards while penalizing those with lower rewards
        cross_entropy = K.log(action_prob) * discounted_rewards
         # Takes the negative sum of cross-entropy to define the loss (policy gradient maximizes this)
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # get action from policy network
    def get_action(self, state):
        # Predicts the probability distribution for each action in the current state using the policy network.
        policy = self.model.predict(state)[0]        
        
        # Randomly selects an action based on the predicted policy distribution to introduce exploration.
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        # Initialize an array to store discounted rewards, matching the shape of the rewards array.
        discounted_rewards = np.zeros_like(rewards)
        # Initialize the running total to accumulate discounted rewards.
        running_add = 0
        
         # Iterate over rewards in reverse order (from the end of the episode to the beginning).
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        # Stores the current state by appending it to the agent's list of states.
        self.states.append(state[0])
        # Adds the current reward to the agent's list of rewards.
        self.rewards.append(reward)
        
        # Creates a one-hot encoded vector representing the chosen action.
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        # Computes the discounted rewards for the episode and normalizes them.
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # Trains the policy network with states, actions, and normalized discounted rewards.
        self.optimizer([self.states, self.actions, discounted_rewards])
        # Resets states, actions, and rewards for the next episode.
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # fresh env
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # update policy neural network for each episode
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")
