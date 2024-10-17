import dataclasses
# 旨在简化数据类的定义，减少样板代码，并提供更好的代码可读性。这对于处理大量数据对象的情况特别有用。
import numpy as np

@dataclasses.dataclass
class VehicularEnvConfig:
    def __init__(self):
        # 道路信息
        self.road_range: int = 600  # 道路长度
        self.road_width: int = 50  # 道路宽度

        # 时间信息
        self.time_slot_start: int = 0
        self.time_slot_end: int = 19
        # 任务信息相关（要处理的任务）
        self.Function_min_task_datasize=4#
        self.Function_max_task_datasize = 6 #
        self.Function_task_computing_resource: float = 300  # 任务计算资源300cycles/bit
        self.Function_min_task_delay: int = 10 # 任务的最小延迟20s
        self.Function_max_task_delay: int = 20  # 任务的最大延迟25s

        # 任务队列相关（每个卸载对象自己产生的任务，即自身到达任务）
        self.min_rsuself_task_number: int = 2    #RSU最小任务个数
        self.max_rsuself_task_number: int = 6  #RSU最大任务个数
        self.min_rsuself_task_datasize: float = 4  # 2 MB 每个任务的最小数据大小
        self.max_rsuself_task_datasize: float = 8  # 4 MB   每个任务的最大数据大小
        self.min_vehicleself_task_number: int = 1    #车辆最小任务个数,用于生成初始任务的个数
        self.max_vehicleself_task_number: int = 2   #车辆最大任务个数,用于生成初始任务的个数
        self.min_vehicleself_task_datasize: float = 1  # 2 MB 每个任务的最小数据大小
        self.max_vehicleself_task_datasize: float = 2  # 4 MB   每个任务的最大数据大小

        # 车辆相关
        self.min_vehicle_speed: int = 40 #车辆行驶的最小速度
        self.max_vehicle_speed: int = 50 #车辆行驶的最大速度
        self.min_vehicle_compute_ability: float =20000  #最小计算能力20000Mcycles/s
        self.max_vehicle_compute_ability: float =25000   #最大计算能力40000Mcycles/s
        self.min_vehicle_energy_consumption: float =1*(10**(-10))  #最小能耗j/cycle
        self.max_vehicle_energy_consumption: float =2*(10**(-10))  #最大能耗
        self.vehicle_number = 10    #车辆个数
        # self.seed = 1    #随机种子
        self.min_vehicle_y_initial_location:float =0    #y坐标最小值
        self.max_vehicle_y_initial_location: float =50  #y坐标最大值
        self.vehicle_x_initial_location:list=[0,self.road_range]#初始车辆的x坐标初始值

        # RSU相关
        self.rsu_number = 3  #RSU的个数
        self.min_rsu_compute_ability: float = 25000 # 最小计算能力25000Mcycles/s
        self.max_rsu_compute_ability: float = 30000  # 最大计算能力30000Mcycles/s
        self.min_rsu_energy_consumption: float =1*(10**(-9))  #最小能耗
        self.max_rsu_energy_consumption: float =2 *(10**(-9))  #最大能耗

        # 通信相关
        self.rsu_range: int =200  # RSU通信距离200m
        self.vehicle_range: int = 150  # 车辆通信距离100m
        self.r2v_B: float = 2  # R2V带宽：20Mbps
        self.v2v_B: float = 4 # V2V带宽:40Mbps
        self.rsu_p: float = 5  # RSU发射功率：50w
        self.vehicle_p: float = 1  # 车发射功率： 10w
        self.w: float = 0.001  # 噪声功率𝜔：0.001 W/Hz
        self.k: float = 30  # 固定损耗𝐾：20-40db取30
        self.theta: int = 2  # 路径损耗因子𝜎：2-6取2
        self.r2r_rate: float =0.5  # r2r传输速率
        self.r2c_rate: float = 0.2  # r2c传输速率：0.2mb/s
        self.cloud_compute_ability:float=30000  #cloud计算能力15000Mcycles/s
        self.cloud_energy_consumption: float =3*(10**(-9))

        self.node_weight = 0.5  # 计算节点权重时所用到的系数
        self.max_diatance=400
        self.max_sum_tasks =50
        self.collaborateDRL_reward_weight = 0.9
        self.RSUDRL_reward_weight = 0.9# 计算奖励时时延和能耗的系数
        self.lstm_step = 3  # LSTM模型的步长
        self.history_data_number = 10  # 历史速度集合中元素的个数
        self.effect_size_discount = 0.95  # 计算效应值里的折扣因子

        # 惩罚
        self.collaborateDRL_punishment = -1000
        self.RSUDRL_punishment = -500

        # 环境相关
        self.seed = 1    #随机种子，保证初始环境的一致性
        self.max_hop=3#最大朓数

        # RSU决策DRL相关
        self.RSUDRL_action_size = (self.rsu_number + 2) ** 3  # 动作空间
        # 状态空间的最大值
        self.RSUDRL_high = np.array(
            [np.finfo(np.float32).max for _ in range(self.rsu_number + self.vehicle_number )])
        # 状态空间的最小值
        self.RSUDRL_low = np.array([0 for _ in range(self.rsu_number + self.vehicle_number )])
        # 共同协作决策DRL相关
        self.collaborateDRL_action_size = (self.vehicle_number )** 2  # 动作空间
        # 状态空间的最大值
        self.collaborateDRL_high = np.array(
            [np.finfo(np.float32).max for _ in range( self.vehicle_number )])
        # 状态空间的最小值
        self.collaborateDRL_low = np.array(
            [0 for _ in range(self.vehicle_number)])