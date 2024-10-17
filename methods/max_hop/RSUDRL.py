
import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
# print(curr_path)

# sys.path.append(parent_path)  # 添加路径到系统路径
parent_path_1 = os.path.dirname(parent_path)
sys.path.append(parent_path_1)
# print(parent_path)
import dataclasses
import datetime
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical
from env.environment_max_hop import RoadState
from env.config_max_hop import VehicularEnvConfig
from env.utils import plot_rewards,  save_results_1, plot_completion_rate, plot_delay, plot_energy_consumption
import dill as pickle  # 用dill代替pickle,保存参数的
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        # np.arange() 函数返回一个有终点和起点的固定步长的排列
        # 参数个数情况： np.arange() 函数分为一个参数，两个参数，三个参数三种情况
        # 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
        # 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
        # 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数。
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        # np.random, shuffle作用就是重新排序返回一个随机序列作用类似洗牌
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    # 此时抽样，顺序并没有打乱
    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPO:
    def __init__(self, n_states, n_actions, cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous  # 环境是否为连续动作
        self.policy_clip = cfg.policy_clip  #
        self.n_epochs = cfg.n_epochs  #
        self.gae_lambda = cfg.gae_lambda  #
        self.device = cfg.device
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.critic = Critic(n_states, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, state):
        # state = np.array([state])  # 先转成数组再转tensor更高效
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        # 该方法torch.squeeze()用于从张量中删除大小为1的维度。它返回一个新的张量，其中删除了所有尺寸为1的维度。
        # item()方法用于从包含单个元素的张量中获取Python标量值。它通常在张量只有一个元素并且您想要将该元素提取为常规Python值时使用。
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()

        value = torch.squeeze(value).item()
        return action, probs, value

    # t.item()将Tensor变量转换为python标量（int，float等），其中t是一个Tensor变量，只能是标量，转换后dtype与Tensor的dtype一致。
    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                    # 为什么还要discount
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            # 此时抽样，顺序并没有打乱

            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]

                entropy = -dist.entropy().mean()  # 计算熵
                actor_loss = -(torch.min(weighted_probs, weighted_clipped_probs)).mean() - 0.005 * entropy  # 添加熵项
                # actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # self.actor_optimizer.zero_grad()
                # actor_loss.backward()
                # self.actor_optimizer.step()
                # self.critic_optimizer.zero_grad()
                # critic_loss.backward()
                # self.critic_optimizer.step()

                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()


class Config:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "PPO2"  # 算法名称
        self.env_name = 'RSU offloading decision'  # 环境名称
        self.continuous = False  # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 0  # 随机种子，置0则不设置随机种子
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ####################################
        self.batch_size =4   # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.n_epochs = 5
        self.actor_lr = 0.0001  # actor的学习率
        self.critic_lr = 0.0001  # critic的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20 # 策略更新频率
        ################################################################################
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
        self.result_path=curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'
        self.model_path=curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/models/'
        self.save_fig=True
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU




def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = RoadState()  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    if cfg.continuous:
        n_actions = env.action_space.shape[0]  # 动作维度
    else:
        n_actions = env.action_space.n  # 动作维度
    agent = PPO(n_states, n_actions, cfg)  # 创建智能体
    if hasattr(env, 'reset'):  # 检查环境是否具有 reset 方法
        if cfg.seed != 0:  # 设置随机种子
            torch.manual_seed(cfg.seed)
            env.reset(seed=cfg.seed)
            np.random.seed(cfg.seed)

    return env, agent



def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards_plot = []  # 记录所有回合的奖励画图
    ma_rewards_plot = []  # 记录所有回合的滑动平均奖励画图
    train_episodes = []  # 记录所有回合数，用于画图
    delay_plot = []
    ma_delay_plot = []
    energy_consumption_plot = []
    ma_energy_consumption_plot = []
    offloading_vehicle_number_plot = []
    offloading_rsu_number_plot = []
    offloading_cloud_number_plot = []
    completion_rate_plot=[]
    ma_completion_rate_plot = []
    steps = 0

    for i_ep in range(cfg.train_eps):

        train_episodes.append(i_ep)
        state = env.reset()
        done = False
        ep_reward = 0
        ep_delay=0
        ep_energy_consumption=0
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number =0

        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward,dealy ,energy_consumption,done,  offloading_rsu,offloading_vehicle, offloading_cloud, complete = env.step(action)
            steps += 1
            ep_reward += reward
            ep_delay+=dealy
            ep_energy_consumption+=energy_consumption
            agent.memory.push(state, action, prob, val, reward, done)
            if steps % cfg.update_fre == 0:
                agent.update()
            state = state_

            offloading_vehicle_number+=offloading_vehicle
            offloading_rsu_number+=offloading_rsu
            offloading_cloud_number+=offloading_cloud
            complete_number+=complete

        rewards_plot.append(ep_reward)
        delay_plot.append(ep_delay)
        energy_consumption_plot.append(ep_energy_consumption)
        offloading_vehicle_number_plot.append(offloading_vehicle_number)
        offloading_rsu_number_plot.append(offloading_rsu_number)
        offloading_cloud_number_plot.append( offloading_cloud_number)
        completion_rate=complete_number/(VehicularEnvConfig().rsu_number*(VehicularEnvConfig().time_slot_end+1))
        completion_rate_plot.append(completion_rate)
        print("#  episode :{}, steps : {}, rewards : {}, delay : {}, energyconsumption : {}, complete : {}, vehicle : {}, rsu : {}, cloud : {}"
              .format(i_ep+1,steps, ep_reward,ep_delay,ep_energy_consumption,
                      completion_rate,offloading_vehicle_number,offloading_rsu_number,offloading_cloud_number))
        # 检查目录是否存在，如果不存在，则创建它
        if not os.path.exists(cfg.result_path):
            os.makedirs(cfg.result_path)

        # 打开一个TXT文件，如果文件不存在则创建它，并以追加模式打开
        with open(cfg.result_path+'experimental_result.txt', 'a') as file:
            # 写入数据
            file.write("# episode :{}, steps :{}, rewards :{}, delay :{}, energy consumption :{}, completion rate :{}, vehicle :{}, rsu :{}, cloud :{}\n".format(i_ep + 1, steps, ep_reward, ep_delay,
                                                                     ep_energy_consumption,
                                                                     completion_rate, offloading_vehicle_number,
                                                                     offloading_rsu_number, offloading_cloud_number))

        if ma_rewards_plot:
            ma_rewards_plot.append(0.9 * ma_rewards_plot[-1] + 0.1 * ep_reward)
        else:
            ma_rewards_plot.append(ep_reward)

        if ma_delay_plot:
            ma_delay_plot.append(0.9 * ma_delay_plot[-1] + 0.1 * ep_delay)
        else:
            ma_delay_plot.append(ep_delay)

        if ma_energy_consumption_plot:
            ma_energy_consumption_plot.append(0.9 * ma_energy_consumption_plot[-1] + 0.1 * ep_energy_consumption)
        else:
            ma_energy_consumption_plot.append(ep_energy_consumption)


        if ma_completion_rate_plot:
            ma_completion_rate_plot.append(0.9 * ma_completion_rate_plot[-1] + 0.1 * completion_rate)
        else:
            ma_completion_rate_plot.append(completion_rate)


    print('完成训练！')
    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    with open(cfg.model_path+'RSUDRL_parameters.pkl', 'wb') as f:
        pickle.dump(agent, f)

    res_dic_rewards = {'rewards': rewards_plot, 'ma_rewards': ma_rewards_plot}
    res_dic_delay = {'delay': delay_plot, 'ma_delay': ma_delay_plot}
    res_dic_energy_consumption = {'energy_consumption': energy_consumption_plot, 'ma_energy_consumption': ma_energy_consumption_plot}
    res_dic_completion_rate = {'completion_rate': completion_rate_plot,
                               'ma_completion_rate': ma_completion_rate_plot}
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_delay, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_energy_consumption, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='train',
                   path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_delay(res_dic_delay['delay'], res_dic_delay['ma_delay'], cfg, tag="train")
    plot_energy_consumption(res_dic_energy_consumption['energy_consumption'], res_dic_energy_consumption['ma_energy_consumption'], cfg, tag="train")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'],
                         cfg, tag="train")
    env.close()



def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    test_episodes = []
    for i_ep in range(cfg.test_eps):
        test_episodes.append(i_ep)
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.test_eps, ep_reward))
    print('完成训练！')

    return rewards, ma_rewards, test_episodes


if __name__ == "__main__":
    cfg = Config()
    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    # 打开一个TXT文件，以追加模式打开
    with open(cfg.model_path+'algorithm_parameters.txt', 'w') as file:
        # 循环写入每个参数及其值
        for key, value in vars(cfg).items():
            file.write("{}: {}\n".format(key, value))

    enviroment_config=VehicularEnvConfig()
    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)

    # 打开一个文件写入配置信息
    with open(cfg.model_path+'enviromental_parameters.txt', 'w') as file:
        for key, value in enviroment_config.__dict__.items():
            # 如果值是numpy数组，转换为列表以便更好地表示
            if isinstance(value, np.ndarray):
                value = value.tolist()
            file.write(f'{key}: {value}\n')

    # 训练
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)


    # # 测试
    # env, agent = env_agent_config(cfg)
    # agent = torch.load('net.pth')
    # rewards, ma_rewards, test_episodes = test(cfg, env, agent)
    # plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {} for {}".format(
    #     cfg.device, cfg.algo_name, cfg.env_name))
    # plt.xlabel('epsiodes')
    # plt.plot(test_episodes, rewards, label='rewards')
    # plt.plot(test_episodes, ma_rewards, label='ma rewards')
    # plt.legend()
    # plt.show()