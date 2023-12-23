import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Do not modify this program

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.system_states = []
        self.observations_for_ppo = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
    
    def clear(self):
        del self.system_states[:]
        del self.observations_for_ppo[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, obs_dim, action_dim, actor_node, critic_node):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.actor = nn.Sequential(
                        nn.Linear(obs_dim, actor_node),
                        nn.Tanh(),
                        nn.Linear(actor_node, actor_node),
                        nn.Tanh(),
                        nn.Linear(actor_node, action_dim),
                        nn.Softmax(dim=-1)
                    )
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, critic_node),
                        nn.Tanh(),
                        nn.Linear(critic_node, critic_node),
                        nn.Tanh(),
                        nn.Linear(critic_node, 1)
                    )
    
    def act(self, observation_for_ppo, occupy, valid_actions):
        if occupy > 0:
            action=torch.tensor(self.action_dim-1)
            action_logprob=torch.tensor(1)
        else:
            action_probs = self.actor(observation_for_ppo)
            action_probs[valid_actions==0]=0
            dist = Categorical(action_probs)
            action = dist.sample()
            valid_actions[action]=0
            valid_actions[-1]=1
            action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), valid_actions
    
    def evaluate(self, system_state, observation_for_ppo, action):
        action_probs = self.actor(observation_for_ppo)
        dist = Categorical(action_probs)        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        system_state_values = self.critic(system_state)

        return action_logprobs, system_state_values, dist_entropy


class PPO:
    def __init__(self, message_num, channel_num, state_dim, obs_dim, action_dim, actor_node, critic_node, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.message_num = message_num
        self.channel_num = channel_num
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.state_dim, self.obs_dim, self.action_dim, actor_node, critic_node).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic(self.state_dim, self.obs_dim, self.action_dim, actor_node, critic_node).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())       
        self.MseLoss = nn.MSELoss()

    def select_action(self, system_state, observation_for_ppo, occupy, valid_actions):
        with torch.no_grad():
            observation_for_ppo = torch.FloatTensor(observation_for_ppo).to(self.device)
            action, action_logprob, valid_actions = self.policy_old.act(observation_for_ppo, occupy, valid_actions)
        action = action.to(self.device)
        action_logprob = action_logprob.to(self.device)            

        self.buffer.system_states.append(system_state)
        self.buffer.observations_for_ppo.append(observation_for_ppo)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item(), valid_actions

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)           
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_system_states = torch.stack(self.buffer.system_states, dim=0).detach().to(self.device)
        old_observations_for_ppo = torch.stack(self.buffer.observations_for_ppo, dim=0).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_logprobs, system_state_values, dist_entropy = self.policy.evaluate(old_system_states, old_observations_for_ppo, old_actions)
            # match states_for_now_values tensor dimensions with rewards tensor
            system_state_values = torch.squeeze(system_state_values)           
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - system_state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(system_state_values, rewards) - 0.01*dist_entropy            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
    
    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)
   
    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        