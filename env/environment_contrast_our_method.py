from typing import Optional, Union, List, Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置TensorFlow日志级别为ERROR
import tensorflow as tf
import gym
from gym import spaces
from gym.core import RenderFrame, ObsType
from env.datastruct_contrast import VehicleList, RSUList, TimeSlot, Function
import torch
from env.config_contrast import VehicularEnvConfig
from methods.contrast.PPO.collaborateDRL_contrast import get_collaborate_decision

import math
import numpy as np
import copy
import random
import time
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import networkx as nx
class RoadState(gym.Env):
    """updating"""

    def __init__(
            self,
            env_config: Optional[VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None
    ):
        self.config = env_config or VehicularEnvConfig()  # 环境参数
        self.timeslot = time_slot or TimeSlot(start=self.config.time_slot_start, end=self.config.time_slot_end)
        self.rsu_number = self.config.rsu_number
        self.vehicle_number = self.config.vehicle_number
        self.rsu_range = self.config.rsu_range
        self.vehicle_range = self.config.vehicle_range
        self.seed = self.config.seed

        # 车辆与RSU的初始化，此处只是初始化了这两个类
        self.vehicle_list = vehicle_list or VehicleList(
            vehicle_number=self.vehicle_number,
            road_range=self.config.road_range,
            vehicle_speed=self.config.vehicle_speed,
            min_task_number=self.config.min_vehicleself_task_number,
            max_task_number=self.config.max_vehicleself_task_number,
            min_task_datasize=self.config.min_vehicleself_task_datasize,
            max_task_datasize=self.config.max_vehicleself_task_datasize,
            min_vehicle_compute_ability=self.config.min_vehicle_compute_ability,
            max_vehicle_compute_ability=self.config.max_vehicle_compute_ability,
            min_vehicle_energy_consumption=self.config.min_vehicle_energy_consumption,
            max_vehicle_energy_consumption=self.config.max_vehicle_energy_consumption,
            vehicle_x_initial_location=self.config.vehicle_x_initial_location,
            min_vehicle_y_initial_location=self.config.min_vehicle_y_initial_location,
            max_vehicle_y_initial_location=self.config.max_vehicle_y_initial_location,
            history_data_number=self.config.history_data_number,
            seed=self.seed
        )


        self.rsu_list = rsu_list or RSUList(
            rsu_number=self.rsu_number,
            min_rsu_task_number=self.config.min_rsuself_task_number,
            max_rsu_task_number=self.config.max_rsuself_task_number,
            min_rsu_task_datasize=self.config.min_rsuself_task_datasize,
            max_rsu_task_datasize=self.config.max_rsuself_task_datasize,
            min_rsu_compute_ability= self.config.min_rsu_compute_ability,
            max_rsu_compute_ability=self.config.max_rsu_compute_ability,
            min_rsu_energy_consumption=self.config.min_rsu_energy_consumption,
            max_rsu_energy_consumption=self.config.max_rsu_energy_consumption
            # seed: int
        )
        self.action_space = spaces.Discrete(self.config.RSUDRL_action_size)
        self.observation_space = spaces.Box(low=self.config.RSUDRL_low, high=self.config.RSUDRL_high, dtype=np.float32)
        # 具体来说，spaces.Discrete(self._config.action_size) 创建了一个离散空间对象，其中 self._config.action_size 指定了该离散空间中可能的状态数量。
        # 因此，这行代码的作用是初始化 action_space 变量，并将其设置为表示可能动作的离散空间。在这个空间中，每个动作都是一个整数，范围从0到 self._config.action_size-1。
        self.state = None
        self.reward = 0
        # self.function = None

    def _state_perception(self) -> np.ndarray:
        """ 这只是一个读取操作，在执行动作之前的队列情况"""
        vehicle_state = [vehicle.get_sum_tasks() for vehicle in self.vehicle_list.vehicle_list]
        rsu_state = [rsu.get_sum_tasks() for rsu in self.rsu_list.rsu_list]

        self.state = np.concatenate([vehicle_state, rsu_state])
        #这边是不是要和模型对应一下
        # np.concatenate()用于将两个或多个数组沿指定的轴（维度）连接在一起，创建一个新的数组。
        return np.array(self.state, dtype=np.float32)


    def _function_generator(self) -> List[Function]:
        """ 产生我们关注的任务 """
        new_function = []

        for i in range(self.rsu_number):
            # random.seed(self.seed + i)
            Function_task_datasize = np.random.uniform(self.config.Function_min_task_datasize,
                                                       self.config.Function_max_task_datasize)
            Function_task_delay = int(
                np.random.uniform(self.config.Function_min_task_delay, self.config.Function_max_task_delay))

            function = Function(Function_task_datasize, self.config.Function_task_computing_resource,
                                Function_task_delay)
            new_function.append(function)

        return new_function

    def _reset_road(self) -> None:
        """ 重置RSU队列，车辆队列 """
        self.vehicle_list = VehicleList(
            vehicle_number=self.vehicle_number,
            road_range=self.config.road_range,
            vehicle_speed=self.config.vehicle_speed,
            min_task_number=self.config.min_vehicleself_task_number,
            max_task_number=self.config.max_vehicleself_task_number,
            min_task_datasize=self.config.min_vehicleself_task_datasize,
            max_task_datasize=self.config.max_vehicleself_task_datasize,
            min_vehicle_compute_ability=self.config.min_vehicle_compute_ability,
            max_vehicle_compute_ability=self.config.max_vehicle_compute_ability,
            min_vehicle_energy_consumption=self.config.min_vehicle_energy_consumption,
            max_vehicle_energy_consumption=self.config.max_vehicle_energy_consumption,
            vehicle_x_initial_location=self.config.vehicle_x_initial_location,
            min_vehicle_y_initial_location=self.config.min_vehicle_y_initial_location,
            max_vehicle_y_initial_location=self.config.max_vehicle_y_initial_location,
            history_data_number=self.config.history_data_number,
            seed=self.seed
        )
        self.rsu_list = RSUList(
            rsu_number=self.rsu_number,
            min_rsu_task_number=self.config.min_rsuself_task_number,
            max_rsu_task_number=self.config.max_rsuself_task_number,
            min_rsu_task_datasize=self.config.min_rsuself_task_datasize,
            max_rsu_task_datasize=self.config.max_rsuself_task_datasize,
            min_rsu_compute_ability= self.config.min_rsu_compute_ability,
            max_rsu_compute_ability=self.config.max_rsu_compute_ability,
            min_rsu_energy_consumption=self.config.min_rsu_energy_consumption,
            max_rsu_energy_consumption=self.config.max_rsu_energy_consumption
            # seed: int
        )

        for i in range(self.config.vehicle_number):
            state = np.random.get_state()
            # 设置种子，生成a
            np.random.seed(self.seed + i)
            np.random.set_state(state)
            x_location = i*100

            self.vehicle_list.vehicle_list[i].change_initial_location(x_location)


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            # 这是一个可选参数，它允许你指定一个随机数种子。种子用于控制伪随机数生成器的行为，如果你想要在每次重置时获得相同的随机状态，可以设置种子。通常用于复现实验结果。如果不提供种子，则默认为 None，表示不使用特定的种子。
            return_info: bool = False,
            # 这是一个布尔值参数，它确定是否要在 reset 方法中返回额外的信息。如果设置为 True，则 reset 方法可能会返回一些关于环境状态的额外信息或元数据，以便在需要时进行分析或记录。如果设置为 False，则只返回环境状态。默认为 False。
            options: Optional[dict] = None,
            # 这是一个可选的字典参数，用于传递其他配置选项。字典可以包含任何其他与 reset 方法相关的配置信息，具体取决于你的应用程序或环境的需求。如果不需要任何额外的配置选项，可以将其设置为 None。
    ):
        self.timeslot.reset()  # 重置时间
        self._reset_road()  # 重置道路
        # self.function = self._function_generator()  # 新任务
        self.state = self._state_perception()  # 读取状态
        # return np.array(self.state, dtype=np.float32),self.function

        return np.array(self.state, dtype=np.float32)



    # 获取所有RSU坐标
    def get_all_rsu_location(self):
        rsu_location = []
        group_size = self.config.road_range // self.config.rsu_number
        road_width_plus_5 = self.config.road_width + 5

        for i in range(self.config.rsu_number):
            group_start = i * group_size
            middle = group_start + group_size // 2
            rsu_location.append((middle, road_width_plus_5))

        return rsu_location



    def predict_speed(self, history_speed, n_steps):
        x, y = list(), list()
        for i in range(len(history_speed)):
            end_ix = i + n_steps
            if end_ix > len(history_speed) - 1:
                break
            seq_x, seq_y = history_speed[i:end_ix], history_speed[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        X = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        n_features = 1
        X = tf.reshape(X, (X.shape[0], X.shape[1], n_features))

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=200, verbose=0)

        x_input = tf.convert_to_tensor(history_speed[-n_steps:], dtype=tf.float32)
        x_input = tf.reshape(x_input, (1, n_steps, n_features))
        yhat = model(x_input, training=False)
        yhat = int(yhat[0, 0])

        return yhat



    #更新所有节点的邻居车辆集合，车辆编号1-10，rsu编号11-13,在数组里车辆是0-9，rsu是10-12,因此后面都是车辆是0-9，rsu是10-12
    def update_neighborhood_vehicle(self):
        # 预测所有车辆的速度，方便后面直接调用
        all_vehicle_speed=[]
        # for l in range(self.config.vehicle_number):
        #     all_vehicle_speed.append(self.predict_speed(self.vehicle_list.vehicle_list[l].get_history_speed_list(),self.config.lstm_step))
        for l in range(self.config.vehicle_number):
            all_vehicle_speed.append(self.vehicle_list.vehicle_list[l].get_speed())
        all_node_history_speed=[]
        all_neighborhood_vehicle = []
        for i in range(self.vehicle_number):
            all_node_history_speed.append(self.vehicle_list.vehicle_list[i].get_history_speed_list())
            all_neighborhood_vehicle.append(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())
        for i in range(self.rsu_number):
            all_neighborhood_vehicle.append(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())
        # return all_node_history_speed,all_neighborhood_vehicle
        # 获取了t-1时隙的所有节点的邻居车辆集合以及t时隙之前所有车辆的历史速度集合，因为此时还没更新邻居车辆集合，所以是t-1时隙的所有节点的邻居车辆集合，但是历史速度集合已经更新
        for i in range(self.vehicle_number+self.rsu_number):#遍历所有节点
            if i<=self.config.vehicle_number-1: #车辆
                if len(all_neighborhood_vehicle[i]) != 0:         #如果车辆i的邻居车辆集合不为空
                    for j in range(len(all_neighborhood_vehicle[i])-1,-1,-1):   #遍历车辆i的邻居车辆集合中的车辆j,此处采用倒叙的方式，避免索引超出限制
                        if all_neighborhood_vehicle[i][j][2]==0:        #如果车辆j是t-1时刻预测保留的直接删除
                            del all_neighborhood_vehicle[i][j]
                        else:                                           #如果车辆j是t-1时刻接收到beacon加入的，则预测一下时刻t是否仍然是邻居车辆
                            predict_speed =all_vehicle_speed[all_neighborhood_vehicle[i][j][0]]
                            # predict_speed=self.predict_speed(all_node_history_speed[all_neighborhood_vehicle[i][j][0]],self.config.lstm_step)   #预测一下车辆j在时刻t的速度
                            loaction=self.vehicle_list.vehicle_list[all_neighborhood_vehicle[i][j][0]].get_location().copy()    #这其实是车辆j在时隙t的真实位置，我们需要的是它在t-1时刻的真实位置
                            last_speed = \
                            self.vehicle_list.vehicle_list[all_neighborhood_vehicle[i][j][0]].get_history_speed_list()[
                                -1]#获取t-1时隙车辆j的速度
                            loaction[0] = loaction[0] - 1 * last_speed  # 获取车辆j在时刻t-1的位置
                            predict_location=[loaction[0]+1*predict_speed,loaction[1]]                                      #得到获取车辆j在时刻t的位置
                            predict_distance = ((self.vehicle_list.vehicle_list[i].get_location()[0] - predict_location[
                                0]) ** 2 + (self.vehicle_list.vehicle_list[i].get_location()[1] - predict_location[
                                1]) ** 2) ** 0.5
                            #计算车辆j和车辆i在时刻t的预测距离
                            if predict_distance  > self.config.vehicle_range-5: #留一点时间传输数据
                                del all_neighborhood_vehicle[i][j]
                            else:
                                all_neighborhood_vehicle[i][j][2]=0   #将标记更改，表示车辆j是在时隙t预测保留的，不是接收到beacon加入的
                                all_neighborhood_vehicle[i][j][1]=predict_speed        #预测速度添加是为了计算距离，从而计算数据传输速率的，如果是接收到beacon加入的该速度为实际速度
                                all_neighborhood_vehicle[i][j][3]=predict_distance######################################
            else:     #rsu
                if len(all_neighborhood_vehicle[i]) != 0:       #如果rsu i的邻居车辆集合不为空
                    for j in range(len(all_neighborhood_vehicle[i])-1,-1,-1): #遍历rsu i的邻居车辆集合中的车辆j
                        if all_neighborhood_vehicle[i][j][2]==0:        #如果车辆j是t-1时刻预测保留的直接删除
                            del all_neighborhood_vehicle[i][j]          #如果车辆j是t-1时刻接收到beacon加入的，则预测一下时刻t是否仍然是邻居车辆
                        else:
                            predict_speed = all_vehicle_speed[all_neighborhood_vehicle[i][j][0]]
                            # predict_speed=self.predict_speed(all_node_history_speed[all_neighborhood_vehicle[i][j][0]],self.config.lstm_step)       #预测一下车辆j在时刻t的速度
                            loaction=self.vehicle_list.vehicle_list[all_neighborhood_vehicle[i][j][0]].get_location().copy()#这其实是车辆j在时隙t的真实位置，我们需要的是它在t-1时刻的真实位置
                            last_speed=self.vehicle_list.vehicle_list[all_neighborhood_vehicle[i][j][0]].get_history_speed_list()[-1]
                            loaction[0]=loaction[0]-1*last_speed #获取车辆j在时刻t-1的位置
                            predict_location=[loaction[0]+1*predict_speed,loaction[1]]                                   #得到获取车辆j在时刻t的位置
                            predict_distance = ((self.get_all_rsu_location()[i - self.config.vehicle_number][0] - predict_location[0]) ** 2 + (
                                        self.get_all_rsu_location()[i - self.config.vehicle_number][1] - predict_location[1]) ** 2) ** 0.5
                            # 计算rsu i和车辆j在时刻t的预测距离
                            if predict_distance  > self.config.rsu_range-5:
                                del all_neighborhood_vehicle[i][j]
                            else:
                                all_neighborhood_vehicle[i][j][2]=0
                                all_neighborhood_vehicle[i][j][1] =predict_speed
                                all_neighborhood_vehicle[i][j][3] =predict_distance#####################################
                                # 添加邻居车辆的时候要添加编号，速度，以及标记和距离

        for i in range(self.vehicle_number):     #遍历所有车辆节点
            vehicle_information_i =[i,self.vehicle_list.vehicle_list[i].get_speed(),1,0] #车辆i的邻居车辆信息###############
            if self.vehicle_list.vehicle_list[i].get_beacon_flag()==1:      #在t时刻如果有车辆i发送了beacon
                location_i = self.vehicle_list.vehicle_list[i].get_location()
                for j in range(self.vehicle_number):                   #遍历其它车辆节点
                    flag=0
                    if j != i:
                        location_j=self.vehicle_list.vehicle_list[j].get_location().copy() #获取车辆节点j的位置
                        distance = ((location_i[0] - location_j[0]) ** 2 + (
                                    location_i[1] - location_j[1]) ** 2) ** 0.5
                        vehicle_information_i[3]=distance###############################################################
                        if distance <= self.config.vehicle_range-5:       #如果车辆j收到了车辆i的beacon
                            for k in range(len(all_neighborhood_vehicle[j])): #如果预测保留的节点此刻又接收到了该节点的beacon，直接更新为最新消息
                                if all_neighborhood_vehicle[j][k][0] == i:
                                    flag=1
                                    all_neighborhood_vehicle[j][k]=vehicle_information_i.copy()

                            if flag==0:
                                # vehicle_information_j=[j,self.vehicle_list.vehicle_list[j].get_speed(),1] #####################这边是否要双向添加邻居车辆
                                all_neighborhood_vehicle[j].append(vehicle_information_i.copy())

                                #all_neighborhood_vehicle[j].append(vehicle_information_j) #####################这边是否要双向添加邻居车辆

                for j in range(self.rsu_number):  # 遍历其它RSU节点
                    flag=0
                    location_j=self.get_all_rsu_location()[j]
                    distance = ((location_i[0] - location_j[0]) ** 2 + (
                            location_i[1] - location_j[1]) ** 2) ** 0.5
                    vehicle_information_i[3] = distance#################################################################
                    if distance <= self.config.rsu_range-5:  # 如果rsu j收到了车辆i的beacon
                        for k in range(len(all_neighborhood_vehicle[j+self.config.vehicle_number])):
                            if all_neighborhood_vehicle[j+self.config.vehicle_number][k][0] == i:
                                flag=1
                                all_neighborhood_vehicle[j+self.config.vehicle_number][k] = vehicle_information_i.copy()
                        if flag==0:
                            # vehicle_information_j = [j, self.vehicle_list.vehicle_list[j].get_speed(), 1]
                            all_neighborhood_vehicle[j+self.config.vehicle_number].append(vehicle_information_i.copy())
                             #添加邻居车辆的时候要添加编号，速度，以及标记

        #更新所有邻居车辆集合，上面只是复制体在执行
        for i in range(self.vehicle_number):
            self.vehicle_list.vehicle_list[i].change_neighborhood_vehicle(all_neighborhood_vehicle[i])
        for i in range(self.rsu_number):
            self.rsu_list.rsu_list[i].change_neighborhood_vehicle(all_neighborhood_vehicle[i+self.config.vehicle_number])


    def delete_out_vehicle_indedx_in_neighborhood_vehicle(self,out_vehicle_index_list):
        all_neighborhood_vehicle = []
        for i in range(self.vehicle_number):
            all_neighborhood_vehicle.append(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())
        for i in range(self.rsu_number):
            all_neighborhood_vehicle.append(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())
        for i in range(self.vehicle_number+self.rsu_number):#遍历所有节点
            if i<=self.config.vehicle_number-1: #车辆
                if len(all_neighborhood_vehicle[i]) != 0:         #如果车辆i的邻居车辆集合不为空
                    for j in range(len(all_neighborhood_vehicle[i])-1,-1,-1):   #遍历车辆i的邻居车辆集合中的车辆j,此处采用倒叙的方式，避免索引超出限制
                        for k in range(len(out_vehicle_index_list)):
                            if all_neighborhood_vehicle[i][j][0]==out_vehicle_index_list[k]:
                                del all_neighborhood_vehicle[i][j]
                                break
            else:     #rsu
                if len(all_neighborhood_vehicle[i]) != 0:       #如果rsu i的邻居车辆集合不为空
                    for j in range(len(all_neighborhood_vehicle[i])-1,-1,-1): #遍历rsu i的邻居车辆集合中的车辆j
                        for k in range(len(out_vehicle_index_list)):
                            if all_neighborhood_vehicle[i][j][0]==out_vehicle_index_list[k]:
                                del all_neighborhood_vehicle[i][j]
                                break

        #更新所有邻居车辆集合，上面只是复制体在执行
        for i in range(self.vehicle_number):
            self.vehicle_list.vehicle_list[i].change_neighborhood_vehicle(all_neighborhood_vehicle[i])
        for i in range(self.rsu_number):
            self.rsu_list.rsu_list[i].change_neighborhood_vehicle(all_neighborhood_vehicle[i+self.config.vehicle_number])



    #获取RSU邻居车辆集合的字典，字典里的键值对，rsu的编号已经+10
    def get_rsu_neighbors_dict(self):
        neighbors_dict={}
        for i in range(self.rsu_number):
            node_number = []
            if len(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())!=0:
                for j in range(len(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())):
                    node_number.append(self.rsu_list.rsu_list[i].get_neighborhood_vehicle()[j][0])
            neighbors_dict[i+self.config.vehicle_number]=node_number
        return neighbors_dict

    #获取车辆邻居车辆集合的字典
    def get_vehicle_neighbors_dict(self):
        neighbors_dict={}
        for i in range(self.vehicle_number):
            node_number = []
            if len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())!=0:
                for j in range(len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())):
                    node_number.append(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle()[j][0])
            neighbors_dict[i]=node_number
        return neighbors_dict

    # 获取车辆计算能力字典,
    def get_vehicle_compute_ability_dict(self):
        compute_ability_dict = {}
        for i in range(self.vehicle_number):
            compute_ability = self.vehicle_list.vehicle_list[i].get_compute_ability().copy()
            compute_ability_dict[i] = compute_ability
        return compute_ability_dict

    #获取RSU计算能力字典,rsu编号已经加10
    def get_rsu_compute_ability_dict(self):
        compute_ability_dict = {}
        for i in range(self.rsu_number):
            compute_ability = self.rsu_list.rsu_list[i].get_compute_ability().copy()
            compute_ability_dict[i+self.config.vehicle_number] = compute_ability
        return compute_ability_dict

    # 获取车辆能耗字典
    def get_vehicle_energy_consumption_dict(self):
        energy_consumption_dict = {}
        for i in range(self.vehicle_number):
            energy_consumption = self.vehicle_list.vehicle_list[i].get_energy_consumption().copy()
            energy_consumption_dict[i] = energy_consumption
        return energy_consumption_dict
    # 获取RSU能耗字典,，rsu编号已经加10
    def get_rsu_energy_consumption_dict(self):
        energy_consumption_dict = {}
        for i in range(self.rsu_number):
            energy_consumption = self.rsu_list.rsu_list[i].get_energy_consumption().copy()
            energy_consumption_dict[i+self.config.vehicle_number] = energy_consumption
        return energy_consumption_dict

    # 获取车辆任务量字典
    def get_vehicle_sum_tasks_dict(self):
        sum_tasks_dict = {}
        for i in range(self.vehicle_number):
            sum_tasks = self.vehicle_list.vehicle_list[i].get_sum_tasks().copy()
            sum_tasks_dict[i] = sum_tasks
        return sum_tasks_dict

    # 获取RSU任务量字典,rsu编号已经加10
    def get_rsu_sum_tasks_dict(self):
        sum_tasks_dict = {}
        for i in range(self.rsu_number):
            sum_tasks = self.rsu_list.rsu_list[i].get_sum_tasks().copy()
            sum_tasks_dict[i+self.config.vehicle_number] = sum_tasks
        return sum_tasks_dict



    # 获取邻居车辆的概率字典
    def get_selection_probability_dict(self):
        selection_probability_dict={}
        weight_dict = {}
        diatance_dict={}
        for i in range(self.vehicle_number):
            if len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())!=0:
                for j in range(len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())):
                    node_number=self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle()[j][0]

                    distance=self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle()[j][3]
                    diatance_dict[(i,node_number)]=distance
                    weight_dict[(i, node_number)] = -self.config.node_weight * (distance / self.config.max_diatance) - (
                            1 - self.config.node_weight) * (self.vehicle_list.vehicle_list[
                                                                node_number].get_sum_tasks() / self.config.max_sum_tasks)
                #
        source_weights_sum = {}
        for (source, _), weight in weight_dict.items():
            source_weights_sum.setdefault(source, 0)
            source_weights_sum[source] += math.exp(weight)

        # # 计算每个源节点对应的目标节点的概率
        for (source, target), weight in weight_dict.items():
            probability = math.exp(weight) / source_weights_sum[source]
            selection_probability_dict[(source,target)]=probability
        # return weight_dict,diatance_dict,selection_probability_dict
        return selection_probability_dict
        # return diatance_dict



    #获取某个车辆的效应值

    def calculate_effect_values(self,node, visited=None, memo=None):
        if visited is None:
            visited = set()
        if memo is None:
            memo = {}

        # 若节点已经在之前计算过，则直接返回其价值
        if node in memo:
            return memo[node]

        # 避免环的产生
        if node in visited:
            return (0, 0, 0)
        visited.add(node)

        # 获取节点的初始价值，假设所有未明确给出价值的节点价值为0
        effect_compute_ability = self.get_vehicle_compute_ability_dict().get(node, 0)
        effect_energy_consumption = self.get_vehicle_energy_consumption_dict().get(node, 0)
        effect_sum_tasks=self.get_vehicle_sum_tasks_dict().get(node, 0)

        # 遍历所有邻居
        if node in self.get_vehicle_neighbors_dict():
            for neighbor in self.get_vehicle_neighbors_dict()[node]:
                # 计算邻居贡献的价值
                n_val1, n_val2, n_val3 = self.calculate_effect_values(neighbor,visited, memo)
                prob = self.get_selection_probability_dict().get((node, neighbor), 0)
                contribution1 = prob * n_val1
                contribution2 = prob * n_val2
                contribution3 = prob * n_val3
                effect_compute_ability +=  self.config.effect_size_discount * contribution1
                effect_energy_consumption +=  self.config.effect_size_discount * contribution2
                effect_sum_tasks +=  self.config.effect_size_discount * contribution3

        # 将当前节点计算结果缓存并返回
        memo[node] = ( effect_compute_ability, effect_energy_consumption,  effect_sum_tasks)
        visited.remove(node)
        return memo[node]

    #获取所有车辆的效应值
    def get_effect_values(self):
        node_values = {}
        for node in self.get_vehicle_neighbors_dict():
            node_values[node] = self.calculate_effect_values(node, visited=set(), memo={})
        return node_values


    # 获取所有节点的邻居车辆中的所有传输速率，车辆编号1-10，rsu编号11-13,在数组里车辆是0-9，rsu是10-12,因此后面都是车辆是0-9，rsu是10-12
    def get_transmission_rate_dict(self):
        transmission_rate_dict={}
        #先算车辆节点的
        for i in range(self.vehicle_number):

            if len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())!=0:
                for j in range(len(self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle())):
                    node_number=self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle()[j][0]

                    distance=self.vehicle_list.vehicle_list[i].get_neighborhood_vehicle()[j][3]
                    if distance==0:
                        rate=1000
                    else:
                        rate=self.config.v2v_B* math.log2(1+(self.config.vehicle_p*self.config.k)/(self.config.w * (distance**self.config.theta)))
                    transmission_rate_dict[(i,node_number)]=rate
        #算RSU节点的
        for i in range(self.rsu_number):

            if len(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())!=0:
                for j in range(len(self.rsu_list.rsu_list[i].get_neighborhood_vehicle())):
                    node_number=self.rsu_list.rsu_list[i].get_neighborhood_vehicle()[j][0]

                    distance=self.rsu_list.rsu_list[i].get_neighborhood_vehicle()[j][3]

                    rate=self.config.r2v_B* math.log2(1+(self.config.rsu_p*self.config.k)/(self.config.w * (distance**self.config.theta)))
                    transmission_rate_dict[(i+self.config.vehicle_number, node_number)] = rate
        return transmission_rate_dict

    #动作映射
    def inverse_composite_action(self,action):
        # 计算 z
        r=self.config.rsu_number+2
        z = action // (r * r)
        # 计算 y
        y = (action // r) % r
        # 计算 x
        x = action % r
        # 将结果加上1，使得范围变为1到r
        x += 1
        y += 1
        z += 1
        action_list = [x-1, y-1, z-1]
        return action_list

    #生成任务
    def create_function_task(self):
        new_function=self._function_generator()
        new_task_list=[]
        for i in range(self.config.rsu_number):
            task_list=[]
            task_list.append(i)
            task_list.append(new_function[i].get_task_datasize())
            task_list.append(new_function[i].get_task_computing_resource())
            task_list.append(new_function[i].get_task_delay())
            new_task_list.append(task_list)
        return  new_task_list.copy()


    #获取奖励并且放置任务
    def get_reward_and_function_allocation(self,action,new_task_list):

        action_list=self.inverse_composite_action(action)
        reward_list=[]
        dealy_list=[]
        energy_consumption_list=[]
        success_task=[]#任务成功则+1，失败则+0
        offloading_vehicle = 0
        offloading_rsu = 0
        offloading_cloud = 0
        collaborate_decision=[]######################################################测试用



        task=new_task_list[1].copy()   # [产生该任务的RSU编号，任务大小，任务所需计算资源，任务延迟]
        offloading_number=action_list[1]
        if action_list[1]<=self.rsu_number-1:#如果卸载到RSU上
            offloading_rsu=offloading_rsu+1
            #传输时延
            up_delay=task[1]/self.config.r2r_rate*(abs(1-offloading_number))
            #等待时延
            wait_delay=self.rsu_list.rsu_list[offloading_number].get_sum_tasks()* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.rsu_list.rsu_list[offloading_number].get_compute_ability()* (10 ** 6))
            #计算时延
            process_delay=task[1]* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.rsu_list.rsu_list[offloading_number].get_compute_ability()* (10 ** 6))
            # 总时延成本
            all_delay=up_delay+wait_delay+process_delay
            # 传输能耗
            up_energy_consumption=task[1]/self.config.r2r_rate*(abs(1-offloading_number))*self.config.rsu_p
            # 计算能耗
            process_energy_consumption=task[1]* 8 * 1024 * 1024*self.config.Function_task_computing_resource*self.rsu_list.rsu_list[offloading_number].get_energy_consumption()
            # 总能耗成本
            all_energy_consumption =up_energy_consumption+process_energy_consumption
            # 总成本
            cost=self.config.RSUDRL_reward_weight*all_delay+(1-self.config.RSUDRL_reward_weight)*(all_energy_consumption)
            # print("卸载到RSU上")
            # print(i,action_list[i])
            # print(-cost,all_delay,all_energy_consumption)
            #获取奖励
            # 判断任务延迟是否超出限制
            if all_delay<=task[3]:
                reward = -cost
                reward_list.append(reward)
                dealy_list.append(all_delay)
                energy_consumption_list.append(all_energy_consumption)
                success_task.append(1)
                # 只有成功卸载,才会进行任务放置
                self.rsu_list.rsu_list[offloading_number].get_task_list().add_task_list(task[1])
            else:
                reward = self.config.RSUDRL_punishment
                reward_list.append(reward)
                dealy_list.append(all_delay)
                energy_consumption_list.append(all_energy_consumption)
                success_task.append(0)
            # print(reward)
        #卸载到多辆车辆共同协作,这里的数据都是处理完一个任务就更新的
        elif action_list[1]==self.rsu_number:
            offloading_vehicle = offloading_vehicle + 1
            delay_list=[]
            energy_consumption_list=[]
            current_node=1+self.config.vehicle_number #因为字典里RSU的编号从10开始
            check_last_node =1+self.config.vehicle_number #为了检察训练给出的决策是否有问题
            check_last_node_1 = 1 + self.config.vehicle_number # 为了检察训练给出的决策是否有问题,有2轮检查
            is_not_execute=0
            remaining_tasksize=task[1]
            allocaion_tasksize=[]
            task_information_list = task.copy()
            vehicle_neighbors_dict = self.get_vehicle_neighbors_dict()
            rsu_neighbors_dict = self.get_rsu_neighbors_dict()
            # print("车辆邻居节点")
            # print(vehicle_neighbors_dict)
            # print("RSU邻居节点")
            # print(rsu_neighbors_dict)
            neighbors_dict = {**vehicle_neighbors_dict, **rsu_neighbors_dict}
            vehicle_compute_ability_dict = self.get_vehicle_compute_ability_dict()
            rsu_compute_ability_dict = self.get_rsu_compute_ability_dict()
            compute_ability_dict = {**vehicle_compute_ability_dict, **rsu_compute_ability_dict}
            vehicle_energy_consumption_dict = self.get_vehicle_energy_consumption_dict()
            rsu_energy_consumption_dict = self.get_rsu_energy_consumption_dict()
            energy_consumption_dict = {**vehicle_energy_consumption_dict, **rsu_energy_consumption_dict}
            vehicle_sum_tasks_dict = self.get_vehicle_sum_tasks_dict()
            rsu_sum_tasks_dict = self.get_rsu_sum_tasks_dict()
            sum_tasks_dict = {**vehicle_sum_tasks_dict, **rsu_sum_tasks_dict}
            transmission_rate_dict = self.get_transmission_rate_dict()
            effect_values_dict = self.get_effect_values()
            all_vehicle_number = get_collaborate_decision(task_information_list, neighbors_dict, compute_ability_dict,
                                               energy_consumption_dict, sum_tasks_dict, transmission_rate_dict,
                                               effect_values_dict)
            collaborate_decision.append(all_vehicle_number.copy())######################################################测试用
            # #这边判断的逻辑还得改改
            #
            # for m in range(len(all_vehicle_number)):
            #     if all_vehicle_number[m][1] not in neighbors_dict[check_last_node] and all_vehicle_number[m][1]!=check_last_node:
            #         is_not_execute = is_not_execute +1
            #     check_last_node=all_vehicle_number[m][1]
            # #
            # for n in range(len(all_vehicle_number)):
            #     if all_vehicle_number[n][1]==check_last_node_1 and all_vehicle_number[n][0] != self.vehicle_number:
            #         is_not_execute = is_not_execute + 1
            #     check_last_node_1 = all_vehicle_number[n][1]
            #
            # if is_not_execute!=0:
            #     reward = self.config.RSUDRL_punishment
            #     reward_list.append(reward)
            #     success_task.append(0)
            if all_vehicle_number[-1][0] !=self.config.vehicle_number:
                reward = self.config.RSUDRL_punishment
                reward_list.append(reward)
                success_task.append(0)
                # print(i, action_list[i])
                # print(all_vehicle_number)
                # print("卸载到车辆上失败")
            else:
                for j in range(len(all_vehicle_number)):#[[8,3],[10, 6]]
                    # 当前节点传输数据到下一跳的传输时延
                    if all_vehicle_number[j][0]==self.vehicle_number:
                        up_delay_current_node=0
                    else:
                        up_delay_current_node =remaining_tasksize*(1-all_vehicle_number[j][0]/self.vehicle_number)/transmission_rate_dict[(current_node,all_vehicle_number[j][1])]
                    # 当前节点上的排队等待时延
                    if all_vehicle_number[j][0]==0:
                        wait_delay_current_node =0
                    else:
                        wait_delay_current_node=sum_tasks_dict[current_node]* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(compute_ability_dict[current_node]*(10 ** 6))
                    # 当前节点上的计算时延
                    process_delay_current_node =remaining_tasksize*(all_vehicle_number[j][0]/self.vehicle_number)* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(compute_ability_dict[current_node]*(10 ** 6))
                    # 当前节点上所需总时延，取最大
                    all_delay_current_node=max([wait_delay_current_node+process_delay_current_node,up_delay_current_node])
                    delay_list.append(all_delay_current_node)
                    #当前节点传输数据到下一跳的传输能耗
                    if all_vehicle_number[j][0] == self.vehicle_number:
                        up_energy_consumption_node=0
                    else:
                        if current_node==1+self.config.vehicle_number:
                            up_energy_consumption_node =remaining_tasksize*(1-all_vehicle_number[j][0]/self.vehicle_number)/transmission_rate_dict[(current_node,all_vehicle_number[j][1])]*self.config.rsu_p
                        else:
                            up_energy_consumption_node = remaining_tasksize * (
                                        1 - all_vehicle_number[j][0] / self.vehicle_number) / transmission_rate_dict[(
                            current_node, all_vehicle_number[j][1])] * self.config.vehicle_p
                    #当前节点本地计算能耗
                    process_energy_consumption_node=remaining_tasksize*(all_vehicle_number[j][0]/self.vehicle_number)* 8 * 1024 * 1024*self.config.Function_task_computing_resource*energy_consumption_dict[current_node]
                    # 当前节点上所需总能耗
                    all_energy_consumption_current_node=up_energy_consumption_node + process_energy_consumption_node
                    energy_consumption_list.append(all_energy_consumption_current_node)
                    #添加任务放置的任务
                    allocaion_tasksize.append(remaining_tasksize*(all_vehicle_number[j][0]/self.vehicle_number))
                    #当前节点更新
                    current_node = all_vehicle_number[j][1]
                    #remaining_tasksize更新
                    remaining_tasksize=remaining_tasksize*(1-all_vehicle_number[j][0]/self.vehicle_number)
                all_delay=sum(delay_list)
                all_energy_consumption=sum(energy_consumption_list)
                # 总成本
                cost = self.config.RSUDRL_reward_weight * all_delay + (
                            1 - self.config.RSUDRL_reward_weight) * (all_energy_consumption)
                # print("卸载到车辆上")
                # print(i, action_list[i])
                # print(all_vehicle_number)
                # print(-cost, all_delay, all_energy_consumption)
                # 判断任务延迟是否超出限制
                if all_delay <= task[3]:
                    reward = -cost
                    reward_list.append(reward)
                    dealy_list.append(all_delay)
                    energy_consumption_list.append(all_energy_consumption)
                    success_task.append(1)
                    # 只有成功卸载,才会进行任务放置
                    self.rsu_list.rsu_list[1].get_task_list().add_task_list(allocaion_tasksize[0])
                    for k in range(len(all_vehicle_number)-1):#[[8,3],[10, 6]]
                        self.vehicle_list.vehicle_list[all_vehicle_number[k][1]].get_task_list().add_task_list(allocaion_tasksize[k+1])
                else:
                    reward = self.config.RSUDRL_punishment
                    reward_list.append(reward)
                    dealy_list.append(all_delay)
                    energy_consumption_list.append(all_energy_consumption)
                    success_task.append(0)
                # print(reward)
        #卸载到cloud上
        else:
            offloading_cloud=offloading_cloud+1
            #传输时延
            up_delay=task[1]/self.config.r2c_rate
            #等待时延
            wait_delay=0
            #计算时延
            process_delay=task[1]* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.config.cloud_compute_ability*(10 ** 6))
            # 总时延成本
            all_delay=up_delay+wait_delay+process_delay
            # 传输能耗
            up_energy_consumption=task[1]/self.config.r2c_rate*self.config.rsu_p
            # 计算能耗
            process_energy_consumption=task[1]* 8 * 1024 * 1024*self.config.Function_task_computing_resource*self.config.cloud_energy_consumption
            # 总能耗成本
            all_energy_consumption =up_energy_consumption+process_energy_consumption
            # 总成本
            cost=self.config.RSUDRL_reward_weight*all_delay+(1-self.config.RSUDRL_reward_weight)*(all_energy_consumption)
            # print("卸载到cloud上")
            # print(i, action_list[i])
            # print(-cost,all_delay,all_energy_consumption)
            # 判断任务延迟是否超出限制
            if all_delay<=task[3]:
                reward = -cost
                reward_list.append(reward)
                dealy_list.append(all_delay)
                energy_consumption_list.append(all_energy_consumption)
                success_task.append(1)
                # 只有成功卸载,才会进行任务放置,Cloud计算能力较强，默认没有排队等候时延
            else:
                reward = self.config.RSUDRL_punishment
                reward_list.append(reward)
                dealy_list.append(all_delay)
                energy_consumption_list.append(all_energy_consumption)
                success_task.append(0)
            # print(reward)
        all_task_rewards=sum(reward_list)
        all_task_dealy=sum(dealy_list)
        all_task_energy_consumption=sum(energy_consumption_list)
        success_task_number=sum(success_task)
        # return all_task_rewards, success_task_number, offloading_rsu, offloading_vehicle, offloading_cloud,collaborate_decision

        return all_task_rewards,all_task_dealy,all_task_energy_consumption ,success_task_number, offloading_rsu, offloading_vehicle, offloading_cloud



#####################################################未测试##############################################################


    # 更新道路状态
    def update_road(self):
        self.timeslot.add_time()  # 当前时隙 now+1
        now = self.timeslot.get_now()

        for i in range(self.vehicle_number):
        # 更新所有车辆的位置
            self.vehicle_list.vehicle_list[i].change_location()
        #更新是否发送信标的beacon
            self.vehicle_list.vehicle_list[i].change_beacon_flag()
        # 更新任务队列和车辆
        for vehicle in self.vehicle_list.get_vehicle_list():
            process_ability = copy.deepcopy(vehicle.get_compute_ability())  # 获取计算能力
            process_ability = (process_ability * (10 ** 6)) / (
                        self.config.Function_task_computing_resource * 8 * 1024 * 1024)
            vehicle.get_task_list().delete_data_list(process_ability)  # 每个时隙都会处理队列里的任务量
            vehicle.get_task_list().add_by_slot(1)  # 每个时隙车辆都会自动生成一个任务

        for rsu in self.rsu_list.get_rsu_list():
            process_ability = copy.deepcopy(rsu.get_compute_ability())  # 获取计算能力
            process_ability=( process_ability*(10**6))/(self.config.Function_task_computing_resource*8*1024*1024)
            rsu.get_task_list().delete_data_list(process_ability)  # 每个时隙都会处理队列里的任务量
            rsu.get_task_list().add_by_slot(2)  # 每个时隙RSU都会自动生成二个任务

        # 给出超出范围的车辆的索引
        out_vehicle_index_list=self.vehicle_list.get_out_vehicle_index()
        #遍历所有车辆和RSU节点，删除邻居车辆里关于该节点的信息
        self.delete_out_vehicle_indedx_in_neighborhood_vehicle(out_vehicle_index_list)
        #更换超出范围车辆的信息
        if len(out_vehicle_index_list)!=0:
            for j in range(len(out_vehicle_index_list)):
                self.vehicle_list.replace_out_vehicle(out_vehicle_index_list[j],random.randint(1, 100)+j+1)
                self.vehicle_list.vehicle_list[out_vehicle_index_list[j]].change_initial_location(0)
        return self.vehicle_list, self.rsu_list

    #step
    def step(self,action):
        # 更新所有节点的邻居车辆集合
        self.update_neighborhood_vehicle()
        new_task_list=self.create_function_task()
        # 奖励值更新
        reward,all_task_dealy,all_task_energy_consumption ,complete_number,offloading_rsu,offloading_vehicle,offloading_cloud = self.get_reward_and_function_allocation(action,new_task_list)
        # done更新
        done = self.timeslot.is_end()
        #更新道路状态
        self.vehicle_list, self.rsu_list = self.update_road()
        # 所有车和RSU的情况，包括生存时间，任务队列等
        obs_vehicle_sum_tasks= [float(vehicle.get_sum_tasks()) for vehicle in self.vehicle_list.get_vehicle_list()]
        obs_rsu_sum_tasks = [float(rsu.get_sum_tasks()) for rsu in self.rsu_list.get_rsu_list()]
        obs_vehicle_history_speed=[vehicle.get_history_speed_list() for vehicle in self.vehicle_list.get_vehicle_list()]

        self.state = np.array(obs_vehicle_sum_tasks + obs_rsu_sum_tasks, dtype=np.float32)
        return self.state, reward,all_task_dealy,all_task_energy_consumption , done,  offloading_rsu,offloading_vehicle, offloading_cloud, complete_number





    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass





    def test_get_reward(self,task_information_list):

        vehicle_neighbors_dict = self.get_vehicle_neighbors_dict()
        rsu_neighbors_dict = self.get_rsu_neighbors_dict()
        neighbors_dict = {**vehicle_neighbors_dict, **rsu_neighbors_dict}
        vehicle_compute_ability_dict=self.get_vehicle_compute_ability_dict()
        rsu_compute_ability_dict = self.get_rsu_compute_ability_dict()
        compute_ability_dict = {**vehicle_compute_ability_dict, **rsu_compute_ability_dict}
        vehicle_energy_consumption_dict=self.get_vehicle_energy_consumption_dict()
        rsu_energy_consumption_dict = self.get_rsu_energy_consumption_dict()
        energy_consumption_dict = {**vehicle_energy_consumption_dict, **rsu_energy_consumption_dict}
        vehicle_sum_tasks_dict=self.get_vehicle_sum_tasks_dict()
        rsu_sum_tasks_dict = self.get_rsu_sum_tasks_dict()
        sum_tasks_dict = {**vehicle_sum_tasks_dict, **rsu_sum_tasks_dict}
        transmission_rate_dict=self.get_transmission_rate_dict()
        effect_values_dict=self.get_effect_values()
        action_=get_collaborate_decision(task_information_list,neighbors_dict,compute_ability_dict,energy_consumption_dict,sum_tasks_dict,transmission_rate_dict,effect_values_dict)
        return action_




# # # 测试初始值
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_location())
# print(a.update_neighborhood_vehicle())
# print(a.vehicle_list.get_vehicle_list()[0].get_task_list().delete_data_list())
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].change_speed())
#     print(a.vehicle_list.get_vehicle_list()[i].change_location())
#     print(a.vehicle_list.get_vehicle_list()[i].change_history_speed_list())

# #
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# print(a._state_perception())

# # 测试生成的任务
# for i in range(a.rsu_number):
#     print(a._function_generator()[i].get_task_datasize())
#     print(a._function_generator()[i].get_task_computing_resource())
#     print(a._function_generator()[i].get_task_delay())
# 测试LSTM
# a=RoadState()
# a.reset()
# b=[15,25,35,45,55,65,75,85,95]
# c=a.predict_speed(b,2)
# print(c)
#
# # 测试邻居车辆更新
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())
# print(a.get_transmission_rate_dict())
# print("第二次更新")
# a.update_neighborhood_vehicle()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
# print(a.get_neighbors_dict())



# # 测试邻居车辆更新
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())
# print("更换车辆")
# out_vehicle_index_list=[4,6]
# a.delete_out_vehicle_indedx_in_neighborhood_vehicle(out_vehicle_index_list)
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
#
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())
# if len(out_vehicle_index_list) != 0:
#     for j in range(len(out_vehicle_index_list)):
#         a.vehicle_list.replace_out_vehicle(out_vehicle_index_list[j],random.randint(1, 100)+j+1)
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
#
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())





# # 测试各个字典
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_location())
# print(a.get_all_rsu_location())
# for i in range(6):
#     a.update_road()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_location())

# print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_neighborhood_vehicle_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_neighborhood_vehicle_data())
# print("RSU邻居车辆集合字典")
# print(a.get_rsu_neighbors_dict())
# print("RSU计算能力字典")
# print(a.get_rsu_compute_ability_dict())
# print("RSU能耗字典")
# print(a.get_rsu_energy_consumption_dict())
# print("RSU任务量字典")
# print(a.get_rsu_sum_tasks_dict())
#
# print("车辆邻居车辆集合字典")
# print(a.get_vehicle_neighbors_dict())
# print("车辆计算能力字典")
# print(a.get_vehicle_compute_ability_dict())
# print("车辆能耗字典")
# print(a.get_vehicle_energy_consumption_dict())
# print("车辆任务量字典")
# print(a.get_vehicle_sum_tasks_dict())
# print("车辆下一跳选择概率字典")
# print(a.get_selection_probability_dict())
# print("传输速率字典")
# print(a.get_transmission_rate_dict())

#
# # # 测试更新效应值
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# a.update_neighborhood_vehicle()
# print(a.get_selection_probability_dict())
# print(a.get_vehicle_neighbors_dict())
# print(a.get_effect_values())


# 测试位置,历史速度变化
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_number):
#     print(a.vehicle_list.vehicle_list[i].get_speed())
#     print(a.vehicle_list.vehicle_list[i].get_history_speed_list())
#
# for i in range(a.vehicle_number):
#     a.vehicle_list.vehicle_list[i].change_history_speed_list()
#     a.vehicle_list.vehicle_list[i].change_speed()
#
# print("#############################################")
# for i in range(a.vehicle_number):
#     print(a.vehicle_list.vehicle_list[i].get_speed())
#     print(a.vehicle_list.vehicle_list[i].get_history_speed_list())



# 测试传输速率
# a=RoadState()
# a.reset()
# # for i in range(a.vehicle_list.get_vehicle_number()):
# #     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# # for j in range(a.rsu_list.get_rsu_number()):
# #     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# # print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())
# print(a.get_transmission_rate())
#
# #
# # # # # # #
# # # # # # # 测试collaborateDRL
# a=RoadState()
# a.reset()
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# # a.update_neighborhood_vehicle()
# # a.update_neighborhood_vehicle()
# # a.update_neighborhood_vehicle()
# # a.update_neighborhood_vehicle()
# # a.update_neighborhood_vehicle()
# print(a.get_vehicle_neighbors_dict())
# print(a.get_rsu_neighbors_dict())
# print(a.get_transmission_rate_dict())
# new_task_list=a.create_function_task()
# task_information_list = new_task_list[0]
# print(task_information_list)
# print(a.test_get_reward(task_information_list))
#

#
# # # 测试生成任务,获取奖励，放置任务
# a=RoadState()
# a.reset()
# # print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
# # # for i in range(4):
# # #
# # #     for i in range(a.vehicle_number):
# # #         a.vehicle_list.vehicle_list[i].change_location()
# # #         a.vehicle_list.vehicle_list[i].change_speed()
# # # a.update_neighborhood_vehicle()
# #
# #
# # print("RSU邻居车辆集合字典")
# # print(a.get_rsu_neighbors_dict())
# # print("RSU计算能力字典")
# # print(a.get_rsu_compute_ability_dict())
# # print("RSU能耗字典")
# # print(a.get_rsu_energy_consumption_dict())
# # print("RSU任务量字典")
# # print(a.get_rsu_sum_tasks_dict())
# # print("车辆邻居车辆集合字典")
# # print(a.get_vehicle_neighbors_dict())
# # print("车辆计算能力字典")
# # print(a.get_vehicle_compute_ability_dict())
# # print("车辆能耗字典")
# # print(a.get_vehicle_energy_consumption_dict())
# # print("车辆任务量字典")
# # print(a.get_vehicle_sum_tasks_dict())
# # print("车辆下一跳选择概率字典")
# # print(a.get_selection_probability_dict())
# # #
# new_task_list=a.create_function_task()
# print("生成的任务为信息为：")
# print(new_task_list)
# print("假设动作为：")#124[4,4,4],51[1,0,2],93[3,3,3]
# action=93
# print(action)
#
# print("采取动作返回信息")
# print(a.get_reward_and_function_allocation(action, new_task_list))
# #
# print("RSU任务量字典")
# print(a.get_rsu_sum_tasks_dict())
# print("车辆任务量字典")
# print(a.get_vehicle_sum_tasks_dict())


# # 测试随机生成的任务
# a=RoadState()
# a.reset()
#
# # a.update_neighborhood_vehicle()
# print(a.step(93))
# # a.update_neighborhood_vehicle()
# print(a.step(93))
# # a.update_neighborhood_vehicle()
# print(a.step(93))



# 各种测试
# a=RoadState()
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# print("更新邻居车辆集合")
# a.update_neighborhood_vehicle()
# a.update_neighborhood_vehicle()
#
# print(a. get_transmission_rate_dict())







##############权重那边的分母上的最大值要找个合适的，用于统一归一化###############
##############时延于能耗要不要归一化#######################################