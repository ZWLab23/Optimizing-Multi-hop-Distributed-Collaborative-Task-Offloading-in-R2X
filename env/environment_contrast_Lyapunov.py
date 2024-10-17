from typing import Optional, Union, List, Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置TensorFlow日志级别为ERROR
import gym
from gym import spaces
from gym.core import RenderFrame, ObsType
from env.datastruct_contrast import VehicleList, RSUList, TimeSlot, Function
import torch
from env.config_contrast import VehicularEnvConfig
import numpy as np
import math
import numpy as np
import copy
import random
import networkx as nx

class LyapunovModel(gym.Env):
    """updating"""

    def __init__(
            self,
            stability_tag: str,
            flow_tag: str,
            env_config: Optional[VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None
    ):
        self.config = env_config or VehicularEnvConfig()  # 环境参数
        self.timeslot = time_slot or TimeSlot(start=self.config.time_slot_start, end=self.config.time_slot_end)
        self.stability = stability_tag
        self.flow_tag = flow_tag

        self.rsu_number = self.config.rsu_number
        self.vehicle_number = 3
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

        # 定义动作和状态空间
        action_low = np.zeros(7)
        #数组的长度是 self.vehicle_number + self.rsu_number + 1,该数组的所有元素被初始化为0
        action_high = np.ones(7)
        # 数组的长度是 self.vehicle_number + self.rsu_number + 1,该数组的所有元素被初始化为1
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        #这行代码使用OpenAI Gym的spaces.Box类创建了一个动作空间。这个动作空间是一个连续的数值范围，其最低值由 action_low 定义，最高值由 action_high 定义。dtype 参数指定了动作空间中元素的数据类型，这里是 np.float32。
        # 定义状态空间
        observation_low = np.zeros(self.rsu_number + 3)
        self.observation_high = np.concatenate((np.full(self.rsu_number, 1e+6), np.full(3, 1e+6)))
        self.observation_space = spaces.Box(observation_low, self.observation_high, dtype=np.float32)
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

        for i in range(self.vehicle_number):
            state = np.random.get_state()
            # 设置种子，生成a
            np.random.seed(self.seed + i)

            x_location = (i+3)*100

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



    def _task_execute(self) -> List[float]:
        """ 上一个时隙的任务计算 """
        b_tau = []

        for i in range(self.rsu_number):
            process_ability = copy.deepcopy(self.rsu_list.rsu_list[i].get_compute_ability())  # 获取计算能力
            outsize = (process_ability * (10 ** 6)) / (self.config.Function_task_computing_resource * 8 * 1024 * 1024)
            task_completion_amount = min(outsize, self.rsu_list.rsu_list[i].get_sum_tasks())
            b_tau.append(task_completion_amount)

        for i in range(self.vehicle_number):
            process_ability = copy.deepcopy(self.vehicle_list.vehicle_list[i].get_compute_ability())  # 获取计算能力
            outsize = (process_ability * (10 ** 6)) / (self.config.Function_task_computing_resource * 8 * 1024 * 1024)
            task_completion_amount = min(outsize, self.vehicle_list.vehicle_list[i].get_sum_tasks())
            b_tau.append(task_completion_amount)
        return b_tau


    def _get_c_in(self, action: np.ndarray,new_task):
        """ 任务输入拆分 """
        allocated_size = [new_task[1] * a for a in action]
        c_r_in = allocated_size[:self.rsu_number]
        c_v_in = allocated_size[self.rsu_number:(self.rsu_number + self.vehicle_number)]
        c_c_in = allocated_size[-1]
        return c_r_in, c_v_in, c_c_in


    def _take_action(self, c_r_in: List[float], c_v_in: List[float]) -> None:
        """ 如果满足卸载条件，则执行卸载动作 """
        for rsu, rsu_input in zip(self.rsu_list.rsu_list, c_r_in):
            rsu.get_task_list().add_task_list(rsu_input)

        for vehicle, vehicle_input in zip(self.vehicle_list.vehicle_list, c_v_in):
            vehicle.get_task_list().add_task_list(vehicle_input)


    def _tasklist_update(self) -> List[float]:
        """ 保证无关任务进入 """
        a_tau=[]
        for i in range(self.rsu_number):
            data_sizes = np.random.uniform(self.config.min_rsuself_task_datasize, self.config.max_rsuself_task_datasize, 2)
            a_tau.append(sum(data_sizes))
        for i in range(self.vehicle_number):
            data_sizes = np.random.uniform(self.config.min_vehicleself_task_datasize, self.config.max_vehicleself_task_datasize, 1)
            a_tau.append(sum(data_sizes))
        return a_tau

    def _rsu_spent_time(self, c_r_in: List[float]) -> float:
        """ rsu部分最长执行时间 """
        rsu_time = []
        for i in range(self.rsu_number):
            if c_r_in[i]==0:
                rsu_time.append(0)
            else:
                up_delay=c_r_in[i]/self.config.r2r_rate*(abs(i-1))
                wait_delay=self.rsu_list.rsu_list[i].get_sum_tasks()* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.rsu_list.rsu_list[i].get_compute_ability()*(10 ** 6))
                process_delay=c_r_in[i]* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.rsu_list.rsu_list[i].get_compute_ability()*(10 ** 6))
                rsu_time.append(up_delay + wait_delay + process_delay)

        return max(rsu_time)

    def _vehicle_spent_time(self, c_v_in: List[float]) -> float:
        """ rsu部分最长执行时间 """
        vehicle_time = []
        for i in range(self.vehicle_number):
            if c_v_in[i]==0:
                vehicle_time.append(0)
            else:
                rsu_location=self.get_all_rsu_location()[1]
                vehicle_location=self.vehicle_list.vehicle_list[i].get_location()
                distance = ((rsu_location[0] - vehicle_location[0]) ** 2 + (
                        rsu_location[1] - vehicle_location[1]) ** 2) ** 0.5
                if distance == 0:
                    rate = 1000
                else:
                    rate = self.config.r2v_B * math.log2(
                        1 + (self.config.rsu_p * self.config.k) / (self.config.w * (distance ** self.config.theta)))
                up_delay=c_v_in[i]/rate
                wait_delay=self.vehicle_list.vehicle_list[i].get_sum_tasks()* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.vehicle_list.vehicle_list[i].get_compute_ability()*(10 ** 6))
                process_delay=c_v_in[i]* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.vehicle_list.vehicle_list[i].get_compute_ability()*(10 ** 6))
                vehicle_time.append(up_delay + wait_delay + process_delay)

        return max(vehicle_time)



    def _cloud_spent_time(self, c_c_in: float):
        if c_c_in==0:
            cloud_time=0
        else:
            up_delay=c_c_in/self.config.r2c_rate
            wait_delay=0
            process_delay=c_c_in* 8 * 1024 * 1024*self.config.Function_task_computing_resource/(self.config.cloud_compute_ability*(10 ** 6))
            cloud_time=up_delay+wait_delay+process_delay

        return cloud_time



    def _spent_time(
            self,
            c_r_in: List[float],
            c_v_in: List[float],
            c_c_in: List[float]
    ):
        """ 一个任务的总执行时间 """
        # rsu部分
        rsu_time = self._rsu_spent_time(c_r_in)
        # print("RSU上最大时间:{}".format(rsu_time))
        # vehicle部分
        vehicle_time = self._vehicle_spent_time(c_v_in)
        # print("车辆上最大时间:{}".format(vehicle_time))
        # cloud部分
        cloud_time = self._cloud_spent_time(c_c_in)
        # print(rsu_time,vehicle_time,cloud_time)
        # print("Cloud上最大时间:{}".format(cloud_time))
        return max(rsu_time, vehicle_time, cloud_time)




    def _compute_y(self, a_tau, b_tau, c_r_in: List[float], c_v_in: List[float], tag):
        """ 计算y """
        if tag == "a":
            y_tau_rsu = [max((a_tau[i] + c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
            y_tau_vehicle = [max((a_tau[i + self.rsu_number] + c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in
                             range(self.vehicle_number)]
        else:
            y_tau_rsu = [max((c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
            y_tau_vehicle = [max((c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in range(self.vehicle_number)]
        return y_tau_rsu, y_tau_vehicle

    def _compute_B(self) -> float:
        """ B的计算 """
        B_r = sum((self.config.max_rsuself_task_datasize*2 + self.config.Function_max_task_datasize) ** 2
                  for _ in range(self.rsu_number)) / 2
        B_v = sum((self.config.max_vehicleself_task_datasize + self.config.Function_max_task_datasize) ** 2
                  for _ in range(self.vehicle_number)) / 2
        B_tau = B_r + B_v
        return B_tau

    def _get_Q_tau(self):
        """ 获取时刻t的队列 """
        Q_tau_r = [rsu.get_sum_tasks() for rsu in self.rsu_list.rsu_list]
        Q_tau_v = [vehicle.get_sum_tasks() for vehicle in self.vehicle_list.vehicle_list]
        return Q_tau_r, Q_tau_v

    def _compute_growth(
            self,
            y_tau_rsu: List[float],
            y_tau_vehicle: List[float],
            B_tau: float,
            Q_tau_r: List[float],
            Q_tau_v: List[float]
    ):
        # 计算队伍增长量部分
        growth_r = sum(y_tau_rsu[i] * Q_tau_r[i] for i in range(self.rsu_number))
        growth_v = sum(y_tau_vehicle[j] * Q_tau_v[j] for j in range(self.vehicle_number))
        growth = growth_r + growth_v + B_tau
        return growth

#总的目标第一个框
    def _compute_backlog(
            self, a_tau, b_tau, c_r_in: List[float], c_v_in: List[float], B_tau: float, Q_tau_r: List[float],
            Q_tau_v: List[float]
    ):
        y_tau_rsu = [max((a_tau[i] + c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
        y_tau_vehicle = [max((a_tau[i + self.rsu_number] + c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in
                         range(self.vehicle_number)]
        backlog_r = sum(y_tau_rsu[i] * Q_tau_r[i] for i in range(self.rsu_number))
        backlog_v = sum(y_tau_vehicle[j] * Q_tau_v[j] for j in range(self.vehicle_number))
        backlog = backlog_r + backlog_v + B_tau
        return backlog

#与backlog可以互换
    def _Lyapunov_drift(self, Q_tau_r: List[float], Q_tau_r_: List[float], Q_tau_v: List[float], Q_tau_v_: List[float]):
        """ 增长 """
        drift_r = sum(((Q_tau_r_[i] ** 2) - (Q_tau_r[i] ** 2)) for i in range(self.rsu_number))
        drift_v = sum(((Q_tau_v_[i] ** 2) - (Q_tau_v[i] ** 2)) for i in range(self.vehicle_number))
        lyapunov_drift = (drift_r + drift_v) / 2
        return lyapunov_drift



    def _reward(
            self,
            a_tau: List[float],
            b_tau: List[float],
            delay: float,
            c_r_in: List[float],
            c_v_in: List[float],
            Q_tau_r: List[float],
            Q_tau_v: List[float],
            Q_tau_r_: List[float],#_是t+1时刻
            Q_tau_v_: List[float],
            task_list
    ):
        """ 计算reward """
        success_task=0
        y_tau_rsu, y_tau_vehicle = self._compute_y(a_tau, b_tau, c_r_in, c_v_in, tag=self.stability)
        B_tau = self._compute_B()
        growth = self._compute_growth(y_tau_rsu, y_tau_vehicle, B_tau, Q_tau_r, Q_tau_v)
        backlog = self._compute_backlog(a_tau, b_tau, c_r_in, c_v_in, B_tau, Q_tau_r, Q_tau_v)
        lyapunov_drift = self._Lyapunov_drift(Q_tau_r, Q_tau_r_, Q_tau_v, Q_tau_v_)

        if self.flow_tag == "flow":
            lyapunov_object = growth+ self.config.w_Lyapunov * delay
        else:
            lyapunov_object = lyapunov_drift + self.config.w_Lyapunov * delay

        if delay > task_list[3]:
            reward = self.config.reward_threshold#惩罚
        else:
            reward = - lyapunov_object
            self._take_action(c_r_in, c_v_in)  # 执行卸载动作
            success_task=1

        queue_v = sum(Q_tau_v) / self.vehicle_number
        queue_r = sum(Q_tau_r) / self.rsu_number
        queue = sum(Q_tau_r + Q_tau_v) / (self.rsu_number + self.vehicle_number)
        y_v = sum(y_tau_vehicle) / self.vehicle_number
        y_r = sum(y_tau_rsu) / self.rsu_number
        y = sum(y_tau_rsu + y_tau_vehicle) / (self.rsu_number + self.vehicle_number)

        return reward,success_task, backlog, queue_v, y_v, queue_r, y_r, queue, y




    def step(self, action):
        """ 状态转移 """
        # 产生任务
        task_list=self.create_function_task()[1]
        # print(action)
        c_r_in, c_v_in, c_c_in = self._get_c_in(action,task_list)
        # print("动作:{}".format(action))
        Q_tau_r, Q_tau_v = self._get_Q_tau()  # 获取 t 时刻的队列长度
        b_tau = self._task_execute()  # 计算上个时隙完成的任务数据量
        for i in range(self.rsu_number):
            self.rsu_list.rsu_list[i].get_task_list().delete_data_list(b_tau[i])
        for i in range(self.vehicle_number):
            self.vehicle_list.vehicle_list[i].get_task_list().delete_data_list(b_tau[i+self.rsu_number])
        # self._take_action(c_r_in, c_v_in)  # 执行卸载动作
        delay = self._spent_time(c_r_in=c_r_in,  c_v_in=c_v_in,  c_c_in=c_c_in)  # 计算时间
        # print(delay)
        a_tau = self._tasklist_update()  # 保证其他任务进入
        for i in range(self.rsu_number):
            self.rsu_list.rsu_list[i].get_task_list().add_by_slot_Lyapunov(a_tau[i])
        for i in range(self.vehicle_number):
            self.vehicle_list.vehicle_list[i].get_task_list().add_by_slot_Lyapunov(a_tau[i+self.rsu_number])

        Q_tau_r_, Q_tau_v_ = self._get_Q_tau()  # 获取 t+1 时刻的队列长度
        reward, success_task,backlog, queue_v, y_v, queue_r, y_r, queue, y = self._reward(a_tau, b_tau, delay, c_r_in, c_v_in,
                                                                             Q_tau_r, Q_tau_v, Q_tau_r_, Q_tau_v_,task_list)

        # 车辆更新
        self.vehicle_list.delete_out_vehicle()
        vehicle_number_now = self.vehicle_number - self.vehicle_list.get_vehicle_number()

        # 更新车辆
        if vehicle_number_now > 0:
            # self.vehicle_list.add_stay_vehicle(vehicle_number_now,self.timeslot.get_now())
            self.vehicle_list.add_stay_vehicle(vehicle_number_now, random.randint(1, 100))
        # 状态转移
        self.timeslot.add_time()
        done = self.timeslot.is_end()

        #状态空间
        obs_vehicle_sum_tasks= [float(vehicle.get_sum_tasks()) for vehicle in self.vehicle_list.get_vehicle_list()]
        obs_rsu_sum_tasks = [float(rsu.get_sum_tasks()) for rsu in self.rsu_list.get_rsu_list()]
        self.state = np.array(obs_vehicle_sum_tasks + obs_rsu_sum_tasks, dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward, success_task,backlog, delay, done, queue_v, y_v, queue_r, y_r, queue, y





# state：状态    reward：奖励 backlog：𝜏到𝜏+1全体队列长度的总增长量 delay：时延   done：是否完成  queue_v：车的平均任务量  y_v：车的平均y值
# queue_r：RSU的平均任务量     y_r：RSU的平均y值    queue:所有车和RSU的平均任务量  y：所有车和RSU的平均y值
    def render(self, mode='human'):
        # 不需要渲染，直接返回
        pass

    def close(self):
        pass





# 测试
# a=LyapunovModel(stability_tag="a",flow_tag="flow")
# a.reset()
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_initial_data())
# for j in range(a.rsu_list.get_rsu_number()):
#     print(a.rsu_list.get_rsu_list()[j].get_initial_data())
# for i in range(a.vehicle_list.get_vehicle_number()):
#     print(a.vehicle_list.get_vehicle_list()[i].get_location())
# print(a.get_all_rsu_location())
# print(a._task_execute())
# print(a._tasklist_update())