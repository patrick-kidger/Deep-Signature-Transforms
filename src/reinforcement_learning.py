import candle
import numpy as np
import siglayer
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


def play(env, policy, steps, render=True):
    state_sequence = []
    # We're going to ignore velocity
    position, velocity = env.reset()
    state = np.array([position, 0.])  # Record the timestep
    
    for s in range(steps):
        state_sequence.append(state)
        if render:
            env.render()
            
        state_seq_var = torch.tensor(state_sequence, requires_grad=True, dtype=torch.float).t().unsqueeze(0)

        # select action
        Q = policy(state_seq_var)
        _, action = torch.max(Q, -1)
        action = action.item()

        # take action
        state, _, done, _ = env.step(action)
        position, velocity = state
        state = np.array([position, (s + 1) / steps])  # Record the timestep

        if done:
            break
            
    return np.array(state_sequence), state[0] > 0.5


def train(env, policy, steps, episodes, epsilon=0.2, gamma=0.99, learning_rate=0.001):    
    successes = 0
    max_position = -0.4
    
    loss_history = []
    reward_history = []
    position_history = []
    
    # call policy on example inputs to fully specify model
    # (needed for things using candle.NoInputSpec, such as siglayer.Augment, which is
    # used in the signature policy)
    state_seq_var = torch.tensor([[env.reset()[0], 0.]], dtype=torch.float).t().unsqueeze(0)
    policy(state_seq_var)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # not the same gamma as was passed in kwargs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        pbar.set_description("{} successes".format(successes))
        episode_loss = 0
        episode_reward = 0

        state_sequence = []    
        position, velocity = env.reset()
        state = np.array([position, 0.])
        
        for s in range(steps):
            state_sequence.append(state)
            
            # wrap into batch of length 1
            state_seq_var = torch.tensor(state_sequence, requires_grad=True, dtype=torch.float).t().unsqueeze(0)
            
            # Choose action epsilon-greedily
            Q = policy(state_seq_var)[0]  # unwrap from batch dimension
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, 3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            
            # take action
            state_1, reward, done, _ = env.step(action)
            position_1, velocity_1 = state_1
            state_1 = np.array([position_1, (s + 1) / steps])
            
            # keep track of max position
            max_position = max(position_1, max_position)
                
            # Increase reward for task completion
            reward = position_1
            if position_1 >= 0.5:
                reward += 1
            
            # Find max Q for t+1 state
            state_sequence_1 = state_sequence + [state_1]
            # wrap into batch of length 1
            state_seq_var_1 = torch.tensor(state_sequence_1, dtype=torch.float).t().unsqueeze(0)
            Q1 = policy(state_seq_var_1)[0]  # unwrap from batch dimension
            maxQ1, _ = torch.max(Q1, -1)
            
            # Create target Q value for training the policy
            Q_target = reward + torch.mul(maxQ1, gamma)
            Q_target.detach_()
            
            # Calculate loss
            loss = loss_fn(Q[action], Q_target)
            
            # Update policy
            policy.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            episode_reward += reward
            
            if done:
                if position_1 >= 0.5:
                    # On successful epsisodes, adjust the following parameters
                    epsilon *= .95
                    scheduler.step()
                    successes += 1

                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                position_history.append(position_1)
                break
            else:
                state = state_1

        if episode % 1000 == 0 and episode > 0:
            optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
            # not the same gamma as was passed in kwargs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
                
    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes / episodes * 100))    

    return loss_history, reward_history, position_history


class RandomPolicy(nn.Module):
    def __init__(self, env, **kwargs):
        super(RandomPolicy, self).__init__(**kwargs)
        self.action_space = env.action_space.n
        
    def forward(self, seq):  
        r = np.random.randint(0, self.action_space)
        return torch.eye(self.action_space)[r]

    
class SigPolicy(nn.Module):
    def __init__(self, env, sig_depth=3, **kwargs):
        super(SigPolicy, self).__init__(**kwargs)
        
        channels = 2
        self.augmentation = siglayer.Augment((channels,), 1, include_original=True, include_time=False)
        self.sig = siglayer.Signature(sig_depth)
        self.l1 = nn.Linear(siglayer.sig_dim(channels + 2, sig_depth) + 2, 64)
        self.l2 = nn.Linear(64, env.action_space.n)
        
    def forward(self, seq):
        x = self.augmentation(seq)
        x = self.sig(x)
        x = torch.cat([x, seq[:, :, -1]], dim=-1)
        x = self.l1(x)
        x = F.relu(x)
        return self.l2(x)
    
    
class RNNPolicy(nn.Module):
    def __init__(self, env, **kwargs):
        super(RNNPolicy, self).__init__(**kwargs)
        
        self.rnn = nn.RNN(2, 32, 3, nonlinearity="relu", batch_first=True)
        self.fc1 = nn.Linear(32, env.action_space.n)
        
    def forward(self, seq):
        # seq.shape == (batch, feature, seq)
        seq = seq.transpose(1, 2)
        # seq.shape == (batch, seq, feature)
        out, _ = self.rnn(seq)
        out = out[:, -1, :]
        
        out = self.fc1(out)
        return out
