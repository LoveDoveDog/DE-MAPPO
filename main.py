import os
import glob
import shutil
from datetime import datetime
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from module_ppo import PPO
from module_env import bigenv
#####################################################
print("===========================================")
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("===========================================")
#################-Customized-Area-###################
K_epochs = 5
eps_clip = 0.2
gamma = 0.9
lr_actor = 0.002
lr_critic = 0.002
episode_timestep_length = 400 
total_timestep_length = int(1e6)  
log_episode_interval = 2
printout_episode_interval = 10
update_episode_interval = 4
statis_episode_interval = 10
actor_node=128
critic_node=128
#################-Customized-Area-###################
random_seed=0
message_num=10
channel_num=10
message_arrival_mean_bound=[1,21]
duration_mat_bound=[1,5]
z_constant=10
g_mat_bound=[0.5,2]
request_size=4
v_tradeoff=1
penalty=np.array([1,1,1,1])
#####################################################
torch.manual_seed(random_seed)
np.random.seed(random_seed)
message_arrival_mean=np.random.randint(message_arrival_mean_bound[0],
    message_arrival_mean_bound[1],message_num)
duration_mat=np.random.randint(duration_mat_bound[0],duration_mat_bound[1],
    (message_num,channel_num))
#####################################################
log_timestep_interval = episode_timestep_length * log_episode_interval
printout_timestep_interval = episode_timestep_length * printout_episode_interval
update_timestep_interval = episode_timestep_length * update_episode_interval
statis_timestep_interval = episode_timestep_length * statis_episode_interval
#####################################################
parser = argparse.ArgumentParser()
parser.add_argument('--message_num', default=message_num, type=int)
parser.add_argument('--channel_num', default=channel_num, type=int)
parser.add_argument('--agent_num', default=channel_num, type=int)
parser.add_argument('--message_arrival_mean', default=message_arrival_mean, type=int)
parser.add_argument('--z_constant', default=z_constant, type=float)
parser.add_argument('--duration_mat', default=duration_mat, type=int)
parser.add_argument('--g_mat_bound', default=g_mat_bound, type=float)
parser.add_argument('--request_size', default=request_size, type=int)
parser.add_argument('--v_tradeoff', type=float)
parser.add_argument('--penalty', default=penalty, type=float)
args=parser.parse_args()
###################### training #####################
def train(v_tradeoff, fig_name):
    args.v_tradeoff = v_tradeoff
    ################ training procedure #################
    env = bigenv(args)
    for index_for_large in np.arange(args.agent_num):
        locals()['ppo_agent_'+str(index_for_large).zfill(3)] = PPO(message_num, channel_num, 
            env.state_dim, env.obs_dim, env.action_dim[index_for_large], actor_node, critic_node,
            lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("===========================================")
    log_handler = open('Data_log/log_DE_'+str(v_tradeoff)+'.csv', "w+")
    log_sum_reward = np.zeros(3)
    printout_sum_reward = np.zeros(3)
    statis_sum_reward = np.zeros(3)
    timestep_counter = 0
    episode_counter = 0
    while timestep_counter < total_timestep_length:
        system_state, system_observations = env.reset()
        current_episode_reward = np.zeros(3)
        for _ in np.arange(episode_timestep_length):
            system_action_tem = np.zeros(args.agent_num,dtype=int)
            system_state = torch.FloatTensor(system_state).to(device)
            random_permutation = np.random.permutation(np.arange(args.agent_num))
            # random_permutation = np.arange(args.agent_num)
            valid_actions = np.ones(message_num+1,dtype=int)
            for index_for_large in random_permutation:
                observation_for_ppo = system_observations[index_for_large,:]
                action_for_ppo, valid_actions = locals()[
                    'ppo_agent_'+str(index_for_large).zfill(3)].select_action(system_state, observation_for_ppo, env.occupy[index_for_large], valid_actions)
                system_action_tem[index_for_large]=action_for_ppo
            system_action=system_action_tem.tolist()
            system_state, system_observations, energy, delay, reward = env.step(np.array(system_action))
            for index_for_large in np.arange(args.agent_num):
                locals()['ppo_agent_'+str(index_for_large).zfill(3)
                        ].buffer.rewards.append(reward)
            current_episode_reward += np.array([energy, delay, reward])
            timestep_counter += 1
        log_sum_reward += current_episode_reward
        printout_sum_reward += current_episode_reward
        # update PPO agent
        if (episode_counter+1) % update_episode_interval == 0:
            for index_for_large in np.arange(args.agent_num):
                locals()['ppo_agent_'+str(index_for_large).zfill(3)].update()
        # log file
        if (episode_counter+1) % log_episode_interval == 0:
            log_avg_reward = log_sum_reward / log_timestep_interval
            log_avg_reward = np.around(log_avg_reward, 4) 
            log_handler.write('{},{},{},{}\n'.format(timestep_counter, log_avg_reward[0], log_avg_reward[1], log_avg_reward[2]))
            log_sum_reward = np.zeros(3)
        # print average reward
        if episode_counter+statis_episode_interval>=int(total_timestep_length/episode_timestep_length):
            statis_sum_reward+=current_episode_reward
        if (episode_counter+1) % printout_episode_interval == 0:
            print_avg_reward = printout_sum_reward / printout_timestep_interval
            print_avg_reward = np.around(print_avg_reward, 2)
            print("Episode : {}-{} \t Timestep : {} \t Energy : {} \t Penalty : {} \t Reward : {}".format(
                episode_counter-printout_episode_interval+2, episode_counter+1, timestep_counter, print_avg_reward[0], print_avg_reward[1], print_avg_reward[2]))
            printout_sum_reward = np.zeros(3)
        episode_counter += 1
    log_handler.close()

    data_handler.write('{},{},{},{}\n'.format(v_tradeoff, np.around(statis_sum_reward[0]/statis_timestep_interval, 4),
        np.around(statis_sum_reward[1]/statis_timestep_interval, 4), np.around(statis_sum_reward[2]/statis_timestep_interval, 4)))

    # print total training time
    print("===========================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("===========================================")
    plot_data = np.genfromtxt('Data_log/log_DE_'+str(v_tradeoff)+'.csv', delimiter=',')
    x_data=plot_data[:,0]
    y_data=plot_data[:,-1]
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data)
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.savefig(fig_name+'_'+str(v_tradeoff)+'.png', dpi=300, bbox_inches='tight')


if os.path.exists('Data_log'):
    shutil.rmtree('Data_log')
os.makedirs('Data_log')
current_directory = os.getcwd()
for csv_file in glob.glob(os.path.join(current_directory, '*.csv')):
    os.remove(csv_file)
for fig_file in glob.glob(os.path.join(current_directory, '*.png')):
    os.remove(fig_file)
info_handler = open('A_info_DE.csv', 'w+')
info_handler.write('message_num,{}\n'.format(message_num))
info_handler.write('channel_num,{}\n'.format(channel_num))
info_handler.write('penalty,{}\n'.format(penalty))
info_handler.write('actor_node,{}\n'.format(actor_node))
info_handler.write('critic_node,{}\n'.format(critic_node))
info_handler.close()
data_handler = open('A_data_DE.csv', 'w+')
for v_tradeoff in np.hstack((np.array([0.1]),np.round(np.arange(1,5,0.5),1))):
    train(v_tradeoff, 'v_tradeoff')
data_handler.close()