import sys, os
from typing import Optional, Union, List, Tuple
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径


sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical
from gym import spaces
from env.config import VehicularEnvConfig
import dill as pickle  # 用dill代替pickle,保存参数的


#task_information_list：[产生该任务的RSU编号，任务大小，任务所需计算资源，任务延迟].这边编号不用加10，在这个函数里面会加
#effect_values_dict:[计算能力，能耗，任务之和]
#除了效应值，其它的都是RSU和vehicle所有节点
def get_collaborate_decision(task_information_list,neighbors_dict,compute_ability_dict,energy_consumption_dict,sum_tasks_dict,transmission_rate_dict,effect_values_dict):

    class TaskAllocation(gym.Env):
        """updating"""
        def __init__(self):
            self.task_information_list=task_information_list.copy()
            self.neighbors_dict=neighbors_dict.copy()
            self.compute_ability_dict =compute_ability_dict.copy()
            self.energy_consumption_dict =energy_consumption_dict.copy()
            self.sum_tasks_dict =sum_tasks_dict.copy()
            self.transmission_rate_dict =transmission_rate_dict.copy()
            self.effect_values_dict =effect_values_dict.copy()
            self.hop=0
            self.config = VehicularEnvConfig()  # 环境参数
            self.current_node=task_information_list[0]+self.config.vehicle_number
            self.tasksize=task_information_list[1]
            self.action_space = spaces.Discrete(self.config.collaborateDRL_action_size)
            self.observation_space = spaces.Box(low=self.config.collaborateDRL_low, high=self.config.collaborateDRL_high,
                                                dtype=np.float32)
            # 具体来说，spaces.Discrete(self._config.action_size) 创建了一个离散空间对象，其中 self._config.action_size 指定了该离散空间中可能的状态数量。
            # 因此，这行代码的作用是初始化 action_space 变量，并将其设置为表示可能动作的离散空间。在这个空间中，每个动作都是一个整数，范围从0到 self._config.action_size-1。


            self.state = None
            self.current_node = task_information_list[0]+self.config.vehicle_number
            self.reward = 0

        def reset(self,
            *,
            seed: Optional[int] = None,
            # 这是一个可选参数，它允许你指定一个随机数种子。种子用于控制伪随机数生成器的行为，如果你想要在每次重置时获得相同的随机状态，可以设置种子。通常用于复现实验结果。如果不提供种子，则默认为 None，表示不使用特定的种子。
            return_info: bool = False,
            # 这是一个布尔值参数，它确定是否要在 reset 方法中返回额外的信息。如果设置为 True，则 reset 方法可能会返回一些关于环境状态的额外信息或元数据，以便在需要时进行分析或记录。如果设置为 False，则只返回环境状态。默认为 False。
            options: Optional[dict] = None,
            # 这是一个可选的字典参数，用于传递其他配置选项。字典可以包含任何其他与 reset 方法相关的配置信息，具体取决于你的应用程序或环境的需求。如果不需要任何额外的配置选项，可以将其设置为 None。
            )-> None:
            self.task_information_list=task_information_list.copy()
            self.hop = 1
            self.tasksize = task_information_list[1]
            self.current_node=task_information_list[0]+self.config.vehicle_number
            self.state = [0 for _ in range(self.config.vehicle_number)]
            for i in range(self.config.vehicle_number):
                if i in neighbors_dict[self.current_node]:
                    self.state[i] = 1

            return np.array(self.state, dtype=np.float32)


        # def is_end(self,action):
        #     proportion, offloading_node = self.decomposition_action(action)
        #     #达到最大条数
        #     if self.hop==self.config.max_hop:
        #         return True
        #     #提前全部计算完结束
        #     elif proportion==1:
        #         return True
        #     else:
        #         return False
        #
        #     # # 任务卸载失败
        #     # elif len(neighbors_dict[self.current_node])==0:
        #     #     return True
        #     # elif offloading_node not in neighbors_dict[self.current_node]:
        #     #     return True

        def decomposition_action(self, action):
            # 计算 y 值，因为 y 的范围是 1 到 10，所以直接用整除运算得到 y
            y = action // self.config.vehicle_number
            # 计算 x 值，x 是剩余的部分
            x = action % self.config.vehicle_number
            x += 1
            y += 1
            return x-1, y-1

        def step(self, action):
            proportion,offloading_node=self.decomposition_action(action)

            if self.current_node==task_information_list[0]+self.config.vehicle_number :
                proportion=0

            if self.hop==self.config.max_hop+1:
                proportion=self.config.vehicle_number
                # 当前节点上的排队等待时延
                if proportion==0:
                    wait_delay_current_node =0
                else:
                    wait_delay_current_node = self.sum_tasks_dict[
                                                  self.current_node] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                          (self.compute_ability_dict[self.current_node] * (10 ** 6))
                # 当前节点上的计算时延
                process_delay_current_node = self.tasksize * (
                        proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                             (self.compute_ability_dict[self.current_node] * (10 ** 6))
                # self.sum_tasks_dict[self.current_node] = self.sum_tasks_dict[self.current_node] + self.tasksize * (
                #         1 - proportion / self.config.vehicle_number)  # 更新一下当前节点的上的总任务量,这个其实加不加都行，不太影响
                # 当前节点传输到下一跳的传输时延
                if proportion == self.config.vehicle_number:
                    forward_delay_current_node = 0
                else:
                    forward_delay_current_node = self.tasksize * (1 - proportion / self.config.vehicle_number) / \
                                                 self.transmission_rate_dict[(self.current_node, offloading_node)]
                # 当前节点上的总时延
                All_delay_current_node = max(
                    [wait_delay_current_node + process_delay_current_node, forward_delay_current_node])

                # 下一跳上的效应排队等待时延
                if proportion == self.config.vehicle_number:
                    wait_delay_next_node = 0
                else:
                    wait_delay_next_node = self.effect_values_dict[offloading_node][
                                               2] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                           (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                # 下一跳上的效应计算时延
                process_delay_next_node = self.tasksize * (
                        1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                          (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                # 下一跳上的总时延
                All_delay_next_node = wait_delay_next_node + process_delay_next_node

                # 总时延成本
                All_delay = All_delay_current_node + All_delay_next_node

                # 当前节点上的本地计算能耗
                process_energy_consumption_current_node = self.tasksize * (
                        proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                          self.energy_consumption_dict[self.current_node]
                # 当前节点上的传输到下一跳的传输能耗
                if proportion == self.config.vehicle_number:
                    forward_energy_consumption_current_node = 0
                else:
                    if self.current_node == self.task_information_list[0]:
                        forward_energy_consumption_current_node = self.tasksize * (
                                1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                            self.current_node, offloading_node)] * self.config.rsu_p
                    else:
                        forward_energy_consumption_current_node = self.tasksize * (
                                1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                            self.current_node, offloading_node)] * self.config.vehicle_p
                # 当前节点上的总能耗
                All_energy_consumption_current_node = process_energy_consumption_current_node + forward_energy_consumption_current_node

                # 下一跳上的效应计算能耗
                process_energy_consumption_next_node = self.tasksize * (
                        1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                       self.effect_values_dict[offloading_node][1]
                # 下一跳上的总能耗
                All_energy_consumption_next_node = process_energy_consumption_next_node
                # 总能耗成本
                All_energy_consumption = All_energy_consumption_current_node + All_energy_consumption_next_node

                # 总cost
                cost = self.config.collaborateDRL_reward_weight * All_delay + (
                        1 - self.config.collaborateDRL_reward_weight) * All_energy_consumption
                self.reward = -cost
                done=True

                # 任务量更新
                self.tasksize = (1- proportion / self.config.vehicle_number)* self.tasksize
                #跳数+1
                self.hop=self.hop-1
                 # 当前节点更新
                self.current_node = offloading_node

            else:
                if len(neighbors_dict[self.current_node])==0:

                    # done = True
                    # self.reward = self.config.collaborateDRL_punishment + (
                    #             self.config.max_hop - self.hop) * self.config.collaborateDRL_punishment
                    # # 任务量更新
                    # self.tasksize = (1 - proportion / self.config.vehicle_number) * self.tasksize
                    # # 跳数+1
                    # self.hop = self.hop
                    # # 当前节点更新
                    # self.current_node = offloading_node
                    proportion = self.config.vehicle_number
                    # 当前节点上的排队等待时延
                    if proportion == 0:
                        wait_delay_current_node = 0
                    else:
                        wait_delay_current_node = self.sum_tasks_dict[
                                                      self.current_node] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                  (self.compute_ability_dict[self.current_node] * (10 ** 6))
                    # 当前节点上的计算时延
                    process_delay_current_node = self.tasksize * (
                            proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                 (self.compute_ability_dict[self.current_node] * (10 ** 6))
                    # self.sum_tasks_dict[self.current_node] = self.sum_tasks_dict[self.current_node] + self.tasksize * (
                    #         1 - proportion / self.config.vehicle_number)  # 更新一下当前节点的上的总任务量,这个其实加不加都行，不太影响
                    # 当前节点传输到下一跳的传输时延
                    if proportion == self.config.vehicle_number:
                        forward_delay_current_node = 0
                    else:
                        forward_delay_current_node = self.tasksize * (1 - proportion / self.config.vehicle_number) / \
                                                     self.transmission_rate_dict[(self.current_node, offloading_node)]
                    # 当前节点上的总时延
                    All_delay_current_node = max(
                        [wait_delay_current_node + process_delay_current_node, forward_delay_current_node])

                    # 下一跳上的效应排队等待时延
                    if proportion == self.config.vehicle_number:
                        wait_delay_next_node = 0
                    else:
                        wait_delay_next_node = self.effect_values_dict[offloading_node][
                                                   2] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                               (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                    # 下一跳上的效应计算时延
                    process_delay_next_node = self.tasksize * (
                            1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                              (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                    # 下一跳上的总时延
                    All_delay_next_node = wait_delay_next_node + process_delay_next_node

                    # 总时延成本
                    All_delay = All_delay_current_node + All_delay_next_node

                    # 当前节点上的本地计算能耗
                    process_energy_consumption_current_node = self.tasksize * (
                            proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                              self.energy_consumption_dict[self.current_node]
                    # 当前节点上的传输到下一跳的传输能耗
                    if proportion == self.config.vehicle_number:
                        forward_energy_consumption_current_node = 0
                    else:
                        if self.current_node == self.task_information_list[0]:
                            forward_energy_consumption_current_node = self.tasksize * (
                                    1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                                self.current_node, offloading_node)] * self.config.rsu_p
                        else:
                            forward_energy_consumption_current_node = self.tasksize * (
                                    1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                                self.current_node, offloading_node)] * self.config.vehicle_p
                    # 当前节点上的总能耗
                    All_energy_consumption_current_node = process_energy_consumption_current_node + forward_energy_consumption_current_node

                    # 下一跳上的效应计算能耗
                    process_energy_consumption_next_node = self.tasksize * (
                            1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                           self.effect_values_dict[offloading_node][1]
                    # 下一跳上的总能耗
                    All_energy_consumption_next_node = process_energy_consumption_next_node
                    # 总能耗成本
                    All_energy_consumption = All_energy_consumption_current_node + All_energy_consumption_next_node

                    # 总cost
                    cost = self.config.collaborateDRL_reward_weight * All_delay + (
                            1 - self.config.collaborateDRL_reward_weight) * All_energy_consumption
                    self.reward = -cost+(self.config.max_hop - self.hop) * self.config.collaborateDRL_punishment


                    done = True

                    # 任务量更新
                    self.tasksize = (1 - proportion / self.config.vehicle_number) * self.tasksize
                    # 跳数+1
                    self.hop = self.hop
                    # 当前节点更新
                    self.current_node = offloading_node
                else:
                    # if offloading_node not in neighbors_dict[self.current_node]:
                    if offloading_node not in neighbors_dict[self.current_node]:
                        self.reward=self.config.max_hop*self.config.collaborateDRL_punishment
                        done = True

                        # 任务量更新
                        self.tasksize = self.tasksize
                        #跳数+1
                        self.hop=self.hop
                         # 当前节点更新
                        self.current_node = offloading_node


                    else:
                        # 当前节点上的排队等待时延
                        if proportion == 0:
                            wait_delay_current_node = 0
                        else:

                            wait_delay_current_node = self.sum_tasks_dict[
                                                          self.current_node] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                      (self.compute_ability_dict[self.current_node] * (10 ** 6))
                        # 当前节点上的计算时延
                        process_delay_current_node = self.tasksize * (
                                proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                     (self.compute_ability_dict[self.current_node] * (10 ** 6))
                        # self.sum_tasks_dict[self.current_node] = self.sum_tasks_dict[self.current_node] + self.tasksize * (
                        #         1 - proportion / self.config.vehicle_number)  # 更新一下当前节点的上的总任务量,这个其实加不加都行，不太影响
                        # 当前节点传输到下一跳的传输时延
                        if proportion == self.config.vehicle_number:
                            forward_delay_current_node = 0
                        else:
                            forward_delay_current_node = self.tasksize * (1 - proportion / self.config.vehicle_number) / \
                                                         self.transmission_rate_dict[(self.current_node, offloading_node)]
                        # 当前节点上的总时延
                        All_delay_current_node = max(
                            [wait_delay_current_node + process_delay_current_node, forward_delay_current_node])

                        # 下一跳上的效应排队等待时延
                        if proportion == self.config.vehicle_number:
                            wait_delay_next_node = 0
                        else:
                            wait_delay_next_node = self.effect_values_dict[offloading_node][
                                                       2] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                   (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                        # 下一跳上的效应计算时延
                        process_delay_next_node = self.tasksize * (
                                    1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / \
                                                  (self.effect_values_dict[offloading_node][0] * (10 ** 6))
                        # 下一跳上的总时延
                        All_delay_next_node = wait_delay_next_node + process_delay_next_node

                        # 总时延成本
                        All_delay = All_delay_current_node + All_delay_next_node

                        # 当前节点上的本地计算能耗
                        process_energy_consumption_current_node = self.tasksize * (
                                proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                                  self.energy_consumption_dict[self.current_node]
                        # 当前节点上的传输到下一跳的传输能耗
                        if proportion == self.config.vehicle_number:
                            forward_energy_consumption_current_node = 0
                        else:
                            if self.current_node == self.task_information_list[0]:
                                forward_energy_consumption_current_node = self.tasksize * (
                                        1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                                    self.current_node, offloading_node)] * self.config.rsu_p
                            else:
                                forward_energy_consumption_current_node = self.tasksize * (
                                        1 - proportion / self.config.vehicle_number) / self.transmission_rate_dict[(
                                    self.current_node, offloading_node)] * self.config.vehicle_p
                        # 当前节点上的总能耗
                        All_energy_consumption_current_node = process_energy_consumption_current_node + forward_energy_consumption_current_node

                        # 下一跳上的效应计算能耗
                        process_energy_consumption_next_node = self.tasksize * (
                                1 - proportion / self.config.vehicle_number) * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                                               self.effect_values_dict[offloading_node][1]
                        # 下一跳上的总能耗
                        All_energy_consumption_next_node = process_energy_consumption_next_node
                        # 总能耗成本
                        All_energy_consumption = All_energy_consumption_current_node + All_energy_consumption_next_node

                        # 总cost
                        cost = self.config.collaborateDRL_reward_weight * All_delay + (
                                1 - self.config.collaborateDRL_reward_weight) * All_energy_consumption
                        self.reward = -cost

                        if proportion==self.config.vehicle_number:
                            done = True

                            # 任务量更新
                            self.tasksize = (1 - proportion / self.config.vehicle_number) * self.tasksize
                            # 跳数+1
                            self.hop = self.hop
                            # 当前节点更新
                            self.current_node = offloading_node
                        else:
                            done = False

                            # 任务量更新
                            self.tasksize = (1 - proportion / self.config.vehicle_number) * self.tasksize
                            # 跳数+1
                            self.hop = self.hop+1
                            # 当前节点更新
                            self.current_node = offloading_node

            state=[0 for _ in range(self.config.vehicle_number)]
            for i in range(self.config.vehicle_number):
                if i in neighbors_dict[self.current_node]:
                    state[i]=1
            state = np.array(state, dtype=np.float32)

            return state,self.reward,done,self.tasksize,self.current_node,self.hop



        def render(self, mode='human'):
                # 渲染环境的当前状态
            pass

        def close(self):
            pass

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
            # state = np.array(state)  # 先转成数组再转tensor更高效
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
            self.env_name = 'Cooperative computing'  # 环境名称
            self.continuous = False  # 环境是否为连续动作
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
            self.seed = 0  # 随机种子，置0则不设置随机种子
            self.train_eps = 500  # 训练的回合数
            self.test_eps = 30  # 测试的回合数
            ################################################################################

            ################################## 算法超参数 ####################################
            self.batch_size = 4  # mini-batch SGD中的批量大小
            self.gamma = 0.95  # 强化学习中的折扣因子
            self.n_epochs = 5
            self.actor_lr = 0.001  # actor的学习率
            self.critic_lr = 0.001  # critic的学习率
            self.gae_lambda = 0.95
            self.policy_clip = 0.2
            self.hidden_dim = 512
            self.update_fre = 20   # 策略更新频率
            ################################################################################


    def env_agent_config(cfg):
        ''' 创建环境和智能体
        '''
        env = TaskAllocation()  # 创建环境
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
        # print('开始训练！')
        # print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
        rewards = []  # 记录所有回合的奖励画图
        ma_rewards = []  # 记录所有回合的滑动平均奖励画图
        train_episodes = []  # 记录所有回合数，用于画图
        final_action=[]
        final_tasksizes=0
        min_rewards=-10000
        steps = 0
        for i_ep in range(cfg.train_eps):
            all_action_list = []
            current_nodes = []  # 记录所经过节点
            tasksizes = []  # 记录每个时隙剩余任务量
            # hop_number=0
            train_episodes.append(i_ep)
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, prob, val = agent.choose_action(state)
                all_action_list.append(action)
                state_, reward, done, tasksize,current_node,hop = env.step(action)

                current_nodes.append(current_node)
                tasksizes.append(tasksize)
                # hop_number += 1
                steps += 1
                ep_reward += reward
                agent.memory.push(state, action, prob, val, reward, done)
                if steps % cfg.update_fre == 0:
                    agent.update()
                state = state_

            rewards.append(ep_reward)

            if ep_reward>min_rewards:
                final_action=all_action_list.copy()
                final_tasksizes=tasksizes[-1]
                min_rewards=ep_reward


            for i in range(len(all_action_list)):
                j = all_action_list[i]
                all_action_list[i] = list(action_conversion(j))

            # print(f"回合：{i_ep + 1}，奖励：{ep_reward:.2f}，动作：{all_action_list}所经过节点：{current_nodes}，剩余任务量：{tasksizes}")


        # print('完成训练！')
        return final_action,final_tasksizes


    def action_conversion(action):
        # 计算 y 值，因为 y 的范围是 1 到 10，所以直接用整除运算得到 y
        y = action // len(effect_values_dict)
        # 计算 x 值，x 是剩余的部分
        x = action % len(effect_values_dict)
        x += 1
        y += 1
        return x-1, y - 1


    cfg = Config()
    env, agent = env_agent_config(cfg)
    action_list,remaining_tasksize = train(cfg, env, agent)
    for i in range(len(action_list)):
        j=action_list[i]
        action_list[i]=list(action_conversion(j))
    if remaining_tasksize==0:
        action_list[-1][0]=len(effect_values_dict)
    action_list[0][0] =0
    return action_list
    # return action_list,remaining_tasksize#测试用

