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

            x_location = i*100
            np.random.set_state(state)
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
        self.new_task_list = self.create_function_task()[1]
        # return np.array(self.state, dtype=np.float32),self.function
        neighbor_vehicle_list=[5,6,7]

        return np.array(self.state, dtype=np.float32),neighbor_vehicle_list,self.new_task_list



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

    def get_neighbor_vehicle(self):
        neighbor_vehicle=[]
        for i in range(self.vehicle_number):
            x_location=self.vehicle_list.vehicle_list[i].get_location()[0]
            if x_location<600 and x_location>=300:
                neighbor_vehicle.append(i+self.config.rsu_number)
        return neighbor_vehicle


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
    def get_reward_and_function_allocation(self,action_list,new_task_list):
        # print("卸载编号：",action_list)

        reward_list=[]
        dealy_list=[]
        energy_consumption_list=[]
        success_task=[]#任务成功则+1，失败则+0
        offloading_vehicle = 0
        offloading_rsu = 0
        offloading_cloud = 0

        task=new_task_list.copy()   # [产生该任务的RSU编号，任务大小，任务所需计算资源，任务延迟]
        offloading_number=action_list
        if offloading_number<=self.config.rsu_number-1:
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
            # print("总时延：",all_delay,"总能耗：",all_energy_consumption)
            # 判断任务延迟是否超出限制
            if all_delay <= task[3]:
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
        elif offloading_number>=self.config.rsu_number and offloading_number<=self.config.rsu_number+self.config.vehicle_number-1:
            offloading_vehicle = offloading_vehicle + 1
            rsu_location = self.get_all_rsu_location()[1]
            vehicle_location=self.vehicle_list.vehicle_list[offloading_number-self.config.rsu_number].get_location().copy()
            # print(vehicle_location)
            distance=((rsu_location[0] - vehicle_location[0]) ** 2 + (
                                    rsu_location[1] - vehicle_location[1]) ** 2) ** 0.5
            # print("距离",distance)
            if distance > self.config.rsu_range:
                reward = self.config.RSUDRL_punishment
                reward_list.append(reward)
                dealy_list.append(20)
                energy_consumption_list.append(50)
                success_task.append(0)
            else:
                if distance==0:
                    rate=1000
                else:
                    rate=self.config.r2v_B* math.log2(1+(self.config.rsu_p*self.config.k)/(self.config.w * (distance**self.config.theta)))
                # 传输时延
                up_delay = task[1] / rate
                # 等待时延
                wait_delay = self.vehicle_list.vehicle_list[
                                 offloading_number-self.config.rsu_number].get_sum_tasks() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / (
                                         self.vehicle_list.vehicle_list[offloading_number-self.config.rsu_number].get_compute_ability() * (10 ** 6))
                # 计算时延
                process_delay = task[1] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource / (
                            self.vehicle_list.vehicle_list[
                                 offloading_number-self.config.rsu_number].get_compute_ability() * (10 ** 6))
                # 总时延成本
                all_delay = up_delay + wait_delay + process_delay
                # 传输能耗
                up_energy_consumption = task[1] / rate * self.config.rsu_p
                # 计算能耗
                process_energy_consumption = task[1] * 8 * 1024 * 1024 * self.config.Function_task_computing_resource * \
                                             self.vehicle_list.vehicle_list[
                                                 offloading_number - self.config.rsu_number].get_energy_consumption()
                # 总能耗成本
                all_energy_consumption = up_energy_consumption + process_energy_consumption
                # 总成本
                cost = self.config.RSUDRL_reward_weight * all_delay + (1 - self.config.RSUDRL_reward_weight) * (
                    all_energy_consumption)
                # print("总时延：",all_delay,"总能耗：",all_energy_consumption)
                if all_delay <= task[3]:
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
            # print("总时延：",all_delay,"总能耗：",all_energy_consumption)
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
        all_task_rewards=sum(reward_list)
        all_task_dealy=sum(dealy_list)
        all_task_energy_consumption=sum(energy_consumption_list)
        success_task_number=sum(success_task)


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
        #更换超出范围车辆的信息
        if len(out_vehicle_index_list)!=0:
            for j in range(len(out_vehicle_index_list)):
                self.vehicle_list.replace_out_vehicle(out_vehicle_index_list[j],random.randint(1, 100)+j+1)
                self.vehicle_list.vehicle_list[out_vehicle_index_list[j]].change_initial_location(0)
        #
        # for i in range(self.vehicle_number):
            # print(self.vehicle_list.vehicle_list[i].get_location())
            # print(self.vehicle_list.vehicle_list[i].get_sum_tasks())
        return self.vehicle_list, self.rsu_list

    #step
    def step(self,action,old_task_list):

        task_list=old_task_list
        # 奖励值更新
        reward,all_task_dealy,all_task_energy_consumption ,complete_number,offloading_rsu,offloading_vehicle,offloading_cloud = self.get_reward_and_function_allocation(action,task_list)
        # done更新
        done = self.timeslot.is_end()
        #更新道路状态
        self.vehicle_list, self.rsu_list = self.update_road()

        neighbor_vehicle_list=self.get_neighbor_vehicle()
        # print("车辆位置",neighbor_vehicle_list)

        # 所有车和RSU的情况，包括生存时间，任务队列等
        obs_vehicle_sum_tasks= [float(vehicle.get_sum_tasks()) for vehicle in self.vehicle_list.get_vehicle_list()]
        obs_rsu_sum_tasks = [float(rsu.get_sum_tasks()) for rsu in self.rsu_list.get_rsu_list()]

        self.state = np.array( obs_rsu_sum_tasks+obs_vehicle_sum_tasks , dtype=np.float32)
        new_task_list = self.create_function_task()[1]
        return self.state, neighbor_vehicle_list,reward,all_task_dealy,all_task_energy_consumption , done,  offloading_rsu,offloading_vehicle, offloading_cloud, complete_number,new_task_list





    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass

