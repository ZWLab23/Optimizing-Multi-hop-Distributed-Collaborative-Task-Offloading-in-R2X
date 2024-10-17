import numpy as np
from typing import List # -> List[int] 指示应返回整数列表。
import random

########################################################################
#获取任务属性的类：获取任务信息大小，计算能力：每bit所需转数，任务延迟约束
class Function(object):
    """任务属性及其操作"""
    #就是一个任务的三元数组

    def __init__(
            self,
            Function_task_datasize: float,
            Function_task_computing_resource: float,
            Function_task_delay: int

    ) -> None:
        self._Function_task_datasize = Function_task_datasize   #感知任务的大小
        self._Function_task_computing_resource = Function_task_computing_resource   #感知任务每bit的所需计算资源
        self._Function_task_delay = Function_task_delay #感知任务的延迟
    def get_task_datasize(self) -> float:
        return float(self._Function_task_datasize)

    def get_task_computing_resource(self) -> float:
        return float(self._Function_task_computing_resource)

    def get_task_delay(self) -> float:
        return float(self._Function_task_delay)


########################################################################

#操作队列的类：这个类的构造函数初始化了任务队列的属性，包括任务数量、任务大小范围、和随机数生成的种子
#功能有：获取任务列表，返回节点的总任务量，任务列表增加任务，自己产生任务（随着时间变化），处理任务（随着时间变化）
class TaskList(object):
    """节点上的待执行任务队列属性及操作"""

    def __init__(
            self,
            task_number: int,   #节点上待执行任务的个数
            minimum_datasize: float,   #节点上待执行任务大小的最小值
            maximum_datasize: float  #节点上待执行任务大小的最大值
            # seed: int
    ) -> None:
        self._task_number = task_number
        self._minimum_datasize = minimum_datasize
        self._maximum_datasize = maximum_datasize
        # self._seed=seed


        # 生成每个任务的数据量大小
        # np.random.seed(self._seed)  #设置种子确保每次使用相同的种子值运行代码时，都会得到相同的随机数序列。
        self._datasizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, self._task_number)
        #这一行生成一个数组（self._data_sizes），其中包含了在 _minimum_data_size 和 _maximum_data_size 之间均匀分布的随机浮点数。生成的随机值的数量等于 _task_number。
        self._task_list = [_ for _ in self._datasizes] #列表化

    def get_task_list(self) -> List[float]:
        return self._task_list

    def sum_datasize(self) -> float:
        """返回该节点的总任务量"""
        return sum(self._task_list)

    def add_task_list(self, new_data_size) -> None:
        """如果卸载到该节点，任务队列会增加"""
        self._task_list.append(new_data_size)

    def add_by_slot(self, task_number) -> None:
        """在时间转移中任务队列自动生成的任务"""
        data_sizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, task_number)
        for datasize in data_sizes:
            self._task_list.append(datasize)
            self._task_number += 1

    def add_by_slot_Lyapunov(self, task_size) -> None:
        """在时间转移中任务队列自动生成的任务"""
        self._task_list.append(task_size)

    def delete_data_list(self, process_ability) -> None:
        """在时间转移中对任务队列中的任务进行处理"""
        while True:
            # 如果队列中没有任务
            if len(self._task_list) == 0:
                break
            # 如果队列中有任务
            elif process_ability >= self._task_list[0]:  # 单位时间计算能力大于数据量
                process_ability -= self._task_list[0]
                del self._task_list[0]
            else:  # 单位时间计算能力小于数据量
                self._task_list[0] -= process_ability
                break

########################################################################


########################################################################
#对每一辆车进行操作的类
#功能有：获取位置坐标，判断是否不再存活，获取车辆行驶速度，获取车辆计算能力，获取车辆任务队列，获取车辆任务队列里的任务量之和，
#获取车辆历史速度集合以及更新历史速度，获取邻居车辆集合以及更新邻居车辆，获取各种效应值以及更新效应值
class Vehicle(object):
    """车辆属性及其操作"""

    def __init__(
            self,
            road_range: int,    #马路长度
            vehicle_speed: float,  # 车辆最小行驶速度
            min_task_number: float, #车辆队列中任务最小个数
            max_task_number: float, #车辆队列中任务最大个数
            min_task_datasize: float,  #每个任务大小的最大值
            max_task_datasize: float,  # 每个任务大小的最小值
            min_vehicle_compute_ability: float,  # 最小车辆计算能力
            max_vehicle_compute_ability: float,   #最大车辆计算能力
            min_vehicle_energy_consumption: float, # 最小能耗
            max_vehicle_energy_consumption: float, # 最大能耗
            vehicle_x_initial_location: list,  # 初始x坐标
            min_vehicle_y_initial_location: float,   #初始y坐标最小值
            max_vehicle_y_initial_location: float,# 初始y坐标最大值
            history_data_number : int,#历史速度集合的大小
            seed: int
    ) -> None:
        self._road_range = road_range
        self._seed=seed

        #生成初始y坐标
        self._min_vehicle_y_initial_location = min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location = max_vehicle_y_initial_location \
        ######################################################################################################################################
        # 保存当前的随机状态
        state = np.random.get_state()
        # 设置种子，生成a
        np.random.seed(self._seed)

        self._vehicle_y_initial_location = np.random.randint(self._min_vehicle_y_initial_location, self._max_vehicle_y_initial_location, 1)[0]
        np.random.set_state(state)
        # y坐标
        self._vehicle_y_location=self._vehicle_y_initial_location.copy()  #只复制，不引用
        # 生成初始x坐标，x初始坐标为0
        self._vehicle_x_initial_location=300
        # x坐标
        self._vehicle_x_location = self._vehicle_x_initial_location

        #生成初始速度
        ######################################################################################################################################
        self._vehicle_initial_speed = vehicle_speed # 车辆速度
        #生成速度
        self._vehicle_speed=self._vehicle_initial_speed


        # 车辆计算能力生成
        self._max_compute_ability = max_vehicle_compute_ability
        self._min_compute_ability = min_vehicle_compute_ability
        # np.random.seed(self._seed)
        self._vehicle_compute_ability = np.random.uniform(self._min_compute_ability, self._max_compute_ability, 1)[0]

        # 车辆能耗生成
        self._min_vehicle_energy_consumption=min_vehicle_energy_consumption
        self._max_vehicle_energy_consumption = max_vehicle_energy_consumption
        self._vehicle_energy_consumption=np.random.uniform(self._min_vehicle_energy_consumption, self._max_vehicle_energy_consumption, 1)[0]

        # 车辆任务队列生成
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_datasize = max_task_datasize
        self._min_datasize = min_task_datasize
        # np.random.seed(self._seed)
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number,1)[0]
        self._vehicle_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize)
        # self._vehicle_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize, self._seed)


        #生成初始历史速度集合
        self._history_data_number=history_data_number
        self._vehicle_initial_history_speed_list=[]
        ######################################################################################################################################
        for i in range(self._history_data_number):
            self._vehicle_initial_history_speed_list.append(self._vehicle_speed)

        #生成历史速度集合
        self._vehiclel_history_speed_list =self._vehicle_initial_history_speed_list.copy()

        #生成初始邻居车辆集合，集合中的元素是车辆编号以及预测的速度和是否说收到beacon加入的标记还有他们之间的距离
        self._vehicle_initial_neighborhood_vehicle=[]
        #生成邻居车辆集合
        self._vehicle_neighborhood_vehicle=self._vehicle_initial_neighborhood_vehicle.copy()


        # 生成初始效应计算能力
        self._vehicle_initial_effect_compute_ability =[]     #效应计算能力
        # 生成效应计算能力
        self._vehicle_effect_compute_ability = self._vehicle_initial_effect_compute_ability.copy()  #效应计算能力
        # 生成初始效应能耗
        self._vehicle_initial_effect_energy_consumption =[]
        # 生成效应能耗
        self._vehicle_effect_energy_consumption =self._vehicle_initial_effect_energy_consumption.copy()
        # 生成初始效应任务队列大小
        self._vehicle_initial_effect_sum_tasks = []
        # 生成效应任务队列大小
        self._vehicle_effect_sum_tasks =self._vehicle_initial_effect_sum_tasks.copy()  #效应任务队列大小
        #################################################################################################
        # 生成初始发送beacon标记

        state = np.random.get_state()
        # 设置种子，生成a
        np.random.seed(self._seed)
        self._initial_beacon_flag = np.random.choice([-1, 1])
        np.random.set_state(state)

        # 生成发送beacon标记
        self._beacon_flag= self._initial_beacon_flag.copy()


    # 获取初始信息,包括初始坐标，初始速度，初始历史速度集合
    def get_initial_data(self) -> list:
        data = [[self._vehicle_x_initial_location, self._vehicle_y_initial_location], self._vehicle_initial_speed, self._vehicle_initial_history_speed_list,
                self._vehicle_compute_ability,self._vehicle_energy_consumption,self._vehicle_initial_neighborhood_vehicle,self._vehicle_initial_effect_compute_ability,
                self._vehicle_initial_effect_energy_consumption,self._vehicle_initial_effect_sum_tasks,self._initial_beacon_flag,
                self._vehicle_task_list.sum_datasize()]
        return data


    def get_neighborhood_vehicle_data(self) -> list: #测试用的
        return self._vehicle_neighborhood_vehicle

    #获取计算能力
    def get_compute_ability(self) -> list:
        return self._vehicle_compute_ability

    #获取能耗
    def get_energy_consumption(self) -> list:
        return  self._vehicle_energy_consumption

    # 获取当前t时刻的速度
    def get_speed(self) -> list:
        speed = self._vehicle_speed
        return speed

    # 更新当前t时刻的速度
    def change_speed(self) -> list:
        speed = np.random.randint(self._min_vehicle_speed, self._max_vehicle_speed,1)[0]
        if self._vehicle_speed>0:
                self._vehicle_speed=speed
        else:
                self._vehicle_speed = -speed
        # return self._vehicle_speed

    # 获取当前t时刻的坐标
    def get_location(self) -> list:
        location = [self._vehicle_x_location, self._vehicle_y_location]
        return location
    #更新初始x坐标
    def change_initial_location(self,vehicle_x_location) -> list:
        self._vehicle_x_location = vehicle_x_location
        self._vehicle_y_location = self._vehicle_y_initial_location
        location = [self._vehicle_x_location, self._vehicle_y_location]
        # return location


    #更新当前t时刻的坐标
    def change_location(self) -> list:
        self._vehicle_x_location = self._vehicle_x_location + self._vehicle_speed*1
        self._vehicle_y_location = self._vehicle_y_initial_location
        location = [self._vehicle_x_location, self._vehicle_y_location]
        # return location

    # 获取当前t时刻的历史速度集合
    def get_history_speed_list(self) -> list:
        history_speed_list=self._vehiclel_history_speed_list
        return history_speed_list

    # 更新当前t时刻的历史速度集合
    def change_history_speed_list(self) -> list:
        del self._vehiclel_history_speed_list[0]
        self._vehiclel_history_speed_list.append(self._vehicle_speed)
        # return self._vehiclel_history_speed_list

    # 获取当前t时刻的邻居车辆集合
    def get_neighborhood_vehicle(self) -> list:
        return self._vehicle_neighborhood_vehicle

    # 更新当前t时刻的邻居车辆集合
    def change_neighborhood_vehicle(self,new_neighborhood_vehicle) -> list:
        self._vehicle_neighborhood_vehicle=new_neighborhood_vehicle.copy()
        # return self._vehicle_neighborhood_vehicle

    # # 当前t时刻的邻居车辆集合增加新的车辆
    # def add_neighborhood_vehicle(self,vehicle_number_and_distance) -> list:
    #     self._vehicle_neighborhood_vehicle.append(vehicle_number_and_distance)
    #
    # # 当前t时刻的邻居车辆集合删除某个车辆
    # def del_neighborhood_vehicle(self,vehicle_number_and_distance) -> list:
    #     self._vehicle_neighborhood_vehicle.remove(vehicle_number_and_distance)

    # #获取当前t时刻的效应计算能力
    # def get_effect_compute_ability(self) -> list:
    #     return self._vehicle_effect_compute_ability
    #
    # #更新当前t时刻的效应计算能力
    # def change_effect_compute_ability(self,effect_compute_ability_list) -> list:
    #     self._vehicle_effect_compute_ability +=effect_compute_ability_list
    #     return self._vehicle_effect_compute_ability
    #
    # #获取当前t时刻的效应能耗
    # def get_effect_energy_consumption(self) -> list:
    #     return self._vehicle_effect_energy_consumption
    #
    # #更新当前t时刻的效应能耗
    # def change_effect_energy_consumption(self,effect_energy_consumption_list) -> list:
    #     self._vehicle_effect_energy_consumption += effect_energy_consumption_list
    #     return self._vehicle_effect_energy_consumption
    #
    # #获取当前t时刻的效应任务队列大小
    # def get_effect_sum_tasks(self) -> list:
    #     return self._vehicle_effect_sum_tasks
    #
    # #更新当前t时刻的效应任务队列大小
    # def change_effect_sum_tasks(self,effect_sum_tasks) -> list:
    #     self._vehicle_effect_sum_tasks += effect_sum_tasks
    #     return self._vehicle_effect_sum_tasks

    #判断当前时刻是否发送了beacon
    def get_beacon_flag(self) -> list:
        return self._beacon_flag

    #更新beacon发送标记
    def change_beacon_flag(self) -> list:
        self._beacon_flag=(-1)*self._beacon_flag
        # return self._beacon_flag


    #判断车辆是否超出范围
    def is_out(self) -> bool:
        if self._vehicle_x_location >=self._road_range:
            return True
        else:
            return False

    #判断车辆是否超出范围
    def is_out_Lyapunov(self) -> bool:
        if self._vehicle_x_location >=600:
            return True
        else:
            return False


    #判断车辆是否超出RSU2范围
    # def is_out_RSU2(self) -> bool:
    #     if self._vehicle_x_initial_location == 0:
    #         if self._vehicle_x_location >=self._road_range-50:
    #             return True
    #         else:
    #             return False
    #     else:
    #         if self._vehicle_x_location <=50:
    #             return True
    #         else:
    #             return False


    # 车辆任务队列相关
    #获取车辆上的任务队列
    def get_task_list(self) -> TaskList:
        return self._vehicle_task_list

    # 获取车辆上的任务队列里所有任务的数据量之和
    def get_sum_tasks(self) -> float:
        if len(self._vehicle_task_list.get_task_list()) == 0:  # 车辆上没有任务
            return 0
        else:
            return self._vehicle_task_list.sum_datasize()  # 车辆上有任务



########################################################################
#对所有车辆进行操作的类
# 功能有：获取车辆数量，获取车辆基础信息列表，增加车辆数量，从车辆队列中删除不在范围内的车辆
class VehicleList(object):
    """实现场景中车辆的管理，包括车辆更新、停留时间更新以及任务队列更新"""

    def __init__(
            self,
            vehicle_number: int,  # 车辆个数
            road_range: int,    #马路长度
            vehicle_speed: float,
            min_task_number: float, #车辆队列中任务最小个数
            max_task_number: float, #车辆队列中任务最大个数
            min_task_datasize: float,  #每个任务大小的最大值
            max_task_datasize: float,  # 每个任务大小的最小值
            min_vehicle_compute_ability: float,  # 最小车辆计算能力
            max_vehicle_compute_ability: float,   #最大车辆计算能力
            min_vehicle_energy_consumption: float, # 最小能耗
            max_vehicle_energy_consumption: float, # 最大能耗
            vehicle_x_initial_location: list,  # 初始x坐标
            min_vehicle_y_initial_location: float,   #初始y坐标最小值
            max_vehicle_y_initial_location: float,# 初始y坐标最大值
            history_data_number : int,#历史速度集合的大小
            seed: int
    ) -> None:
        self._seed = seed
        self._vehicle_number = vehicle_number
        self._road_range = road_range
        self._vehicle_speed = vehicle_speed
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_datasize = min_task_datasize
        self._max_datasize = max_task_datasize
        self._min_compute_ability = min_vehicle_compute_ability
        self._max_compute_ability = max_vehicle_compute_ability
        self._min_vehicle_energy_consumption=min_vehicle_energy_consumption
        self._max_vehicle_energy_consumption=max_vehicle_energy_consumption
        self._vehicle_x_initial_location=vehicle_x_initial_location
        self._min_vehicle_y_initial_location= min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location= max_vehicle_y_initial_location
        self._history_data_number=history_data_number
        #车辆基础信息列表，n辆车就有n个信息组
        self.vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                vehicle_speed=self._vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                min_vehicle_energy_consumption=self._min_vehicle_energy_consumption,
                max_vehicle_energy_consumption=self._max_vehicle_energy_consumption,
                vehicle_x_initial_location=self._vehicle_x_initial_location,
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location,
                history_data_number=self._history_data_number,
                seed=self._seed+_
            )
            for _ in range(self._vehicle_number)]


    def get_vehicle_number(self) -> int:
        """获取车辆数量"""
        return self._vehicle_number

    # def get_vehicle_number(self) -> int:
    #     """获取RSU2范围内的车辆数量"""
    #     return self._vehicle_number

    def get_vehicle_list(self) -> List[Vehicle]:
        """获取车辆基础信息队列"""
        return self.vehicle_list

    #Lyapunov
    def add_stay_vehicle(self, new_vehicle_number,time_now) -> None:
        """增加车辆数量"""
        # np.random.seed(self._seed)
        new_vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                ehicle_speed=self.vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                vehicle_x_initial_location=self._vehicle_x_initial_location,  # 初始x坐标
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,  # 初始y坐标最小值
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location, # 初始y坐标最大值
                seed=time_now+_
            )
            for _ in range(new_vehicle_number)]

        self.vehicle_list = self.vehicle_list + new_vehicle_list
        self._vehicle_number += new_vehicle_number


    def replace_out_vehicle(self, index,time_now) -> None:
        """增加车辆数量"""
        # np.random.seed(self._seed)
        new_vehicle = Vehicle(
                road_range=self._road_range,
                vehicle_speed=self._vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                min_vehicle_energy_consumption=self._min_vehicle_energy_consumption,
                max_vehicle_energy_consumption=self._max_vehicle_energy_consumption,
                vehicle_x_initial_location=self._vehicle_x_initial_location,
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location,
                history_data_number=self._history_data_number,
                seed=time_now
            )
        self.vehicle_list[index] = new_vehicle


    def get_out_vehicle_index(self) -> None:
        """从队列中查找不在范围内的车辆编号"""
        is_out_index=[]
        for i in range(len(self.vehicle_list)):
            if self.vehicle_list[i].is_out():
                is_out_index.append(i)
        return is_out_index
        # while i < len(self.vehicle_list):
        #     if len(self.vehicle_list) == 0:#不一定需要判断
        #         pass
        #     elif self.vehicle_list[i].is_out():
        #         del self.vehicle_list[i]
        #         self._vehicle_number -= 1
        #     else:
        #         i += 1
    def delete_out_vehicle(self) -> None:
        """从队列中删除不在范围内的车辆"""
        i = 0
        while i < len(self.vehicle_list):
            if len(self.vehicle_list) == 0:#不一定需要判断
                pass
            elif self.vehicle_list[i].is_out_Lyapunov():
                del self.vehicle_list[i]
                self._vehicle_number -= 1
            else:
                i += 1

########################################################################
#对某个RSU进行操作的类
#功能有：获取RSU计算能力，获取RSU任务队列，获取RSU任务队列上的任务量之和
class RSU(object):
    """RSU"""

    def __init__(
            self,
            min_rsu_task_number: float, #RSU队列中任务最小个数
            max_rsu_task_number: float, #RSU队列中任务最大个数
            min_rsu_task_datasize: float,  # RSU队列中任务大小的最小值
            max_rsu_task_datasize: float,  #RSU队列中任务大小的最大值
            min_rsu_compute_ability: float,  # RSU计算速度的最小值
            max_rsu_compute_ability: float, #RSU计算速度的最大值
            min_rsu_energy_consumption:float,#RSU能耗最小值
            max_rsu_energy_consumption:float #RSU能耗最大值
            # seed: int
    ) -> None:
        # RSU计算速度生成
        self._min_rsu_compute_ability = min_rsu_compute_ability
        self._max_rsu_compute_ability = max_rsu_compute_ability
        # self._seed = seed
        # np.random.seed(self._seed)
        self._rsu_compute_ability = np.random.uniform(self._min_rsu_compute_ability, self._max_rsu_compute_ability, 1)[0]

        # RSU能耗生成
        self._min_rsu_energy_consumption =min_rsu_energy_consumption
        self._max_rsu_energy_consumption = max_rsu_energy_consumption
        self._rsu_energy_consumption = np.random.uniform(self._min_rsu_energy_consumption, self._max_rsu_energy_consumption, 1)[0]
        # RSU任务队列生成
        self._min_rsu_task_number = min_rsu_task_number
        self._max_rsu_task_number = max_rsu_task_number
        self._max_rsu_task_datasize = max_rsu_task_datasize
        self._min_rsu_task_datasize = min_rsu_task_datasize
        # np.random.seed(self._seed)
        self._rsu_task_number = np.random.randint(self._min_rsu_task_number, self._max_rsu_task_number,1)[0]
        self._rsu_task_list = TaskList(self._rsu_task_number, self._min_rsu_task_datasize, self._max_rsu_task_datasize)
        # self._rsu_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize, self._seed)

        #生成初始邻居车辆集合，集合中的元素是车辆编号以及预测的距离
        self._rsu_initial_neighborhood_vehicle=[]
        #生成邻居车辆集合
        self._rsu_neighborhood_vehicle=self._rsu_initial_neighborhood_vehicle.copy()



    def get_initial_data(self) -> list:
            data = [self._rsu_compute_ability,self._rsu_energy_consumption,self._rsu_task_list.sum_datasize()]
            return data

    def get_neighborhood_vehicle_data(self) -> list: #测试用的
        return self._rsu_neighborhood_vehicle

     # 获取RSU计算能力
    def get_compute_ability(self) -> float:
            return self._rsu_compute_ability


     # 获取RSU计算能力
    def get_energy_consumption(self) -> float:
            return self._rsu_energy_consumption


    # 获取RSU任务队列
    def get_task_list(self) -> TaskList:
            return self._rsu_task_list

    # 获取RSU上的任务队列里所有任务的数据量之和
    def get_sum_tasks(self) -> float:
        if len(self._rsu_task_list.get_task_list()) == 0:  # RSU上没有任务
            return 0
        else:
            return self._rsu_task_list.sum_datasize()  # RSU上有任务

    # 获取当前t时刻的邻居车辆集合
    def get_neighborhood_vehicle(self) -> list:
        return self._rsu_neighborhood_vehicle

    def change_neighborhood_vehicle(self,new_neighborhood_vehicle) -> list:
        self._rsu_neighborhood_vehicle=new_neighborhood_vehicle.copy()
        # return self._rsu_neighborhood_vehicle

    # # 当前t时刻的邻居车辆集合增加新的车辆
    # def add_neighborhood_vehicle(self, vehicle_number_and_distance) -> list:
    #     self._rsu_neighborhood_vehicle.append(vehicle_number_and_distance)
    #
    # # 当前t时刻的邻居车辆集合删除某个车辆
    # def del_neighborhood_vehicle(self, vehicle_number_and_distance) -> list:
    #     self._rsu_neighborhood_vehicle.remove(vehicle_number_and_distance)


########################################################################
#对所有RSU进行操作的类
#获取RSU个数，获取RSU上的基础信息组
class RSUList(object):
    """RSU队列管理"""

    def __init__(
            self,
            rsu_number:int, #RSU个数
            min_rsu_task_number: float, #RSU队列中任务最小个数
            max_rsu_task_number: float, #RSU队列中任务最大个数
            min_rsu_task_datasize: float,  # RSU队列中任务大小的最小值
            max_rsu_task_datasize: float,  #RSU队列中任务大小的最大值
            min_rsu_compute_ability: float,  # RSU计算速度的最小值
            max_rsu_compute_ability: float, #RSU计算速度的最大值
            min_rsu_energy_consumption:float,#RSU能耗最小值
            max_rsu_energy_consumption:float #RSU能耗最大值
            # seed: int
    ) -> None:

        # self._seed = seed
        self._rsu_number = rsu_number
        self._min_rsu_task_number = min_rsu_task_number
        self._max_rsu_task_number = max_rsu_task_number
        self._min_rsu_task_datasize = min_rsu_task_datasize
        self._max_rsu_task_datasize = max_rsu_task_datasize
        self._min_rsu_compute_ability = min_rsu_compute_ability
        self._max_rsu_compute_ability = max_rsu_compute_ability
        self._min_rsu_energy_consumption = min_rsu_energy_consumption
        self._max_rsu_energy_consumption = max_rsu_energy_consumption
        # 获取RSU类
        self.rsu_list = [
            RSU(
                min_rsu_task_number=self._min_rsu_task_number,
                max_rsu_task_number=self._max_rsu_task_number,
                min_rsu_task_datasize=self._min_rsu_task_datasize,
                max_rsu_task_datasize=self._max_rsu_task_datasize,
                min_rsu_compute_ability=self._min_rsu_compute_ability,
                max_rsu_compute_ability=self._max_rsu_compute_ability,
                min_rsu_energy_consumption= self._min_rsu_energy_consumption,
                max_rsu_energy_consumption= self._max_rsu_energy_consumption
                # seed: int

            )
            for _ in range(self._rsu_number)]
    #获取RSU个数
    def get_rsu_number(self):
        return self._rsu_number

    #获取RSU的基础信息组
    def get_rsu_list(self):
        return self.rsu_list

########################################################################
#对时隙进行操作的类
class TimeSlot(object):
    """时隙属性及操作"""

    def __init__(self, start: int, end: int) -> None:
        self.start = start  #时间起始间隙
        self.end = end      #时间截止间隙
        self.slot_length = self.end - self.start    #时间间隙长度

        self.now = start    #当前时间间隙定位
        self.reset()    #做一些操作来将对象的属性或状态还原到初始状态

    def __str__(self):
        return f"now time: {self.now}, [{self.start} , {self.end}] with {self.slot_length} slots"

    #随着时间增加
    def add_time(self) -> None:
        """add time to the system"""
        self.now += 1

    #当前是否在时间截止间隙
    def is_end(self) -> bool:
        """check if the system is at the end of the time slots"""
        return self.now >= self.end
    #获取时间间隙长度
    def get_slot_length(self) -> int:
        """get the length of each time slot"""
        return self.slot_length
    #获取当前时间间隙定位
    def get_now(self) -> int:
        return self.now
    #重置
    def reset(self) -> None:
        self.now = self.start
