import numpy as np

class bigenv(object):
    def __init__(self, args):
        self.message_num=args.message_num
        self.channel_num=args.channel_num
        self.message_arrival_mean=args.message_arrival_mean
        self.z_constant=args.z_constant
        self.duration_mat=args.duration_mat
        self.g_mat_bound=args.g_mat_bound
        self.request_size=args.request_size
        self.v_tradeoff=args.v_tradeoff
        self.penalty=args.penalty

        self.g_mat=np.random.uniform(self.g_mat_bound[0],self.g_mat_bound[1],(self.message_num,self.channel_num))
        self.energy_para_mat=self.z_constant*self.duration_mat/self.g_mat

        self.state_dim=self.message_num*self.request_size+self.channel_num+self.message_num*self.channel_num
        self.obs_dim=self.message_num*self.request_size+1+self.message_num
        self.action_dim = self.channel_num*[self.message_num+1]
        
        self.q_mat=np.zeros((self.message_num,self.request_size),dtype=int)
        self.occupy=np.zeros(self.channel_num,dtype=int)

        self.system_state=np.zeros(self.state_dim)
        self.system_observations=np.zeros((self.channel_num,self.obs_dim))

    def step(self, action):
        for index, agent_occupy in enumerate(self.occupy):
            if agent_occupy>0:
                self.occupy[index]-=1

        delay=0
        for message_index in np.arange(self.message_num):
            delay=delay+self.penalty@self.q_mat[message_index,:]

        energy=0
        for channel_index, agent_action in enumerate(action):
            if agent_action!=self.message_num:
                energy=energy+self.v_tradeoff*self.energy_para_mat[agent_action,channel_index]
                self.q_mat[agent_action,:]=0
                self.occupy[channel_index]=self.duration_mat[agent_action,channel_index]-1
        new_arrival=np.random.poisson(self.message_arrival_mean)
        self.q_mat=np.hstack((new_arrival.reshape(-1,1),self.q_mat[:,:-2],(self.q_mat[:,-2]+self.q_mat[:,-1]).reshape(-1,1)))

        self.g_mat=np.random.uniform(self.g_mat_bound[0],self.g_mat_bound[1],(self.message_num,self.channel_num))
        self.energy_para_mat=self.z_constant*self.duration_mat/self.g_mat

        self.system_state=np.hstack((self.q_mat.ravel(),self.occupy.ravel(),self.g_mat.ravel()))
        for channel_index in np.arange(self.channel_num):
            self.system_observations[channel_index,:]=np.hstack((self.q_mat.ravel(),self.occupy[channel_index],self.g_mat[:,channel_index].ravel()))

        return self.system_state, self.system_observations, -energy/self.v_tradeoff, -delay, -energy-delay

    def reset(self):
        self.q_mat=np.zeros((self.message_num,self.request_size),dtype=int)
        self.occupy=np.zeros(self.channel_num,dtype=int)

        self.system_state=np.zeros(self.state_dim)
        self.system_observations=np.zeros((self.channel_num,self.obs_dim))
        
        return self.system_state, self.system_observations

