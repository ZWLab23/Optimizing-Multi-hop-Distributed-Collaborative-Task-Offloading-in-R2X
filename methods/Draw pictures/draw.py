import sys, os
import numpy as np
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
print(curr_path)
import matplotlib.pyplot as plt
import seaborn as sns

from env.utils import plot_PPO_rewards,plot_PPO_completion_ratio,plot_Rewards
from env.utils import plot_PPO_rewards_lr,plot_PPO_completion_ratio_lr,plot_PPO_delay_lr,plot_PPO_energy_consumption_lr
from env.utils import plot_contrast_delay,plot_contrast_completion_ratio
from env.utils import plot_tasksize_rewards,plot_tasksize_completion_ratio,plot_tasksize_delay,plot_tasksize_energy_consumption,plot_tasksize_delay_line_chart,plot_tasksize_energy_consumption_line_chart,plot_tasksize_four_dimensional
from env.utils import plot_tasksize_rewards_box_diagram,plot_tasksize_completion_ratio_box_diagram,plot_tasksize_delay_box_diagram,plot_tasksize_energy_consumption_box_diagram
from env.utils import plot_computing_resource_rewards,plot_computing_resource_completion_ratio,plot_computing_resource_delay,plot_computing_resource_energy_consumption
from env.utils import plot_computing_resource_rewards_box_diagram,plot_computing_resource_completion_ratio_box_diagram,plot_computing_resource_delay_box_diagram,plot_computing_resource_energy_consumption_box_diagram
from env.utils import plot_vehicle_number_rewards,plot_vehicle_number_completion_ratio,plot_vehicle_number_delay,plot_vehicle_number_energy_consumption
from env.utils import plot_vehicle_number_delay_box_diagram,plot_vehicle_number_energy_consumption_box_diagram
from env.utils import plot_vehicle_number_delay_and_energy_consumption_bar_chart,plot_vehicle_number_completion_ratio_box_diagram
from env.utils import plot_vehicle_range_rewards,plot_vehicle_range_completion_ratio,plot_vehicle_range_delay,plot_vehicle_range_energy_consumption
from env.utils import plot_vehicle_range_completion_ratio_box_diagram,plot_vehicle_range_delay_and_energy_consumption_bar_chart
from env.utils import plot_vehicle_speed_rewards,plot_vehicle_speed_completion_ratio,plot_vehicle_speed_delay,plot_vehicle_speed_energy_consumption
from env.utils import plot_vehicle_speed_completion_ratio_box_diagram,plot_vehicle_speed_delay_and_energy_consumption_bar_chart
from env.utils import plot_hob_number_rewards,plot_hob_number_delay,plot_hob_number_energy_consumption
from env.utils import plot_PPO_rewards_beacon_cycle,plot_PPO_completion_ratio_beacon_cycle,plot_PPO_delay_beacon_cycle,plot_PPO_energy_consumption_beacon_cycle,plot_beacon_cycle_delay_and_energy_consumption_bar_chart
from env.utils import plot_beacon_cycle_delay_bar_chart,plot_beacon_cycle_energy_consumption_bar_chart
#绘制绘制不同学习率PPO奖励收敛图
PPO_train_rewards_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_rewards.npy")
PPO_train_rewards_1.tolist()
PPO_train_rewards_1=PPO_train_rewards_1[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_ma_rewards.npy")
PPO_train_ma_rewards_1.tolist()
PPO_train_ma_rewards_1=PPO_train_ma_rewards_1[:200]

PPO_train_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_completion_rate.npy")
PPO_train_completion_ratio_1.tolist()
PPO_train_completion_ratio_1=PPO_train_completion_ratio_1[:200]
PPO_train_ma_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_1.tolist()
PPO_train_ma_completion_ratio_1=PPO_train_ma_completion_ratio_1[:200]

PPO_train_delay_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_delay.npy")
PPO_train_delay_1.tolist()
PPO_train_delay_1=PPO_train_delay_1[:200]
PPO_train_ma_delay_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_ma_delay.npy")
PPO_train_ma_delay_1.tolist()
PPO_train_ma_delay_1=PPO_train_ma_delay_1[:200]

PPO_train_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_energy_consumption.npy")
PPO_train_energy_consumption_1.tolist()
PPO_train_energy_consumption_1=PPO_train_energy_consumption_1[:200]
PPO_train_ma_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.001/20240429-095432/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_1.tolist()
PPO_train_ma_energy_consumption_1=PPO_train_ma_energy_consumption_1[:200]





PPO_train_rewards_2 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0005/20240503-220155/results/train_rewards.npy")
PPO_train_rewards_2.tolist()
PPO_train_rewards_2=PPO_train_rewards_2[:200]
PPO_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0005/20240503-220155/results/train_ma_rewards.npy")
PPO_train_ma_rewards_2.tolist()
PPO_train_ma_rewards_2=PPO_train_ma_rewards_2[:200]

PPO_train_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0005/20240503-220155/results/train_completion_rate.npy")
PPO_train_completion_ratio_2.tolist()
PPO_train_completion_ratio_2=PPO_train_completion_ratio_2[:200]
PPO_train_ma_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/convergence graph//0.0005/20240503-220155/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_2.tolist()
PPO_train_ma_completion_ratio_2=PPO_train_ma_completion_ratio_2[:200]

PPO_train_delay_2 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0005/20240503-220155/results/train_delay.npy")
PPO_train_delay_2.tolist()
PPO_train_delay_2=PPO_train_delay_2[:200]
PPO_train_ma_delay_2 = np.load("C:/Users/23928/Desktop/result/convergence graph//0.0005/20240503-220155/results/train_ma_delay.npy")
PPO_train_ma_delay_2.tolist()
PPO_train_ma_delay_2=PPO_train_ma_delay_2[:200]

PPO_train_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0005/20240503-220155/results/train_energy_consumption.npy")
PPO_train_energy_consumption_2.tolist()
PPO_train_energy_consumption_2=PPO_train_energy_consumption_2[:200]
PPO_train_ma_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/convergence graph//0.0005/20240503-220155/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_2.tolist()
PPO_train_ma_energy_consumption_2=PPO_train_ma_energy_consumption_2[:200]





PPO_train_rewards_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_rewards.npy")
PPO_train_rewards_4.tolist()
PPO_train_rewards_4=PPO_train_rewards_4[:200]
PPO_train_ma_rewards_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_rewards.npy")
PPO_train_ma_rewards_4.tolist()
PPO_train_ma_rewards_4=PPO_train_ma_rewards_4[:200]

PPO_train_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_completion_rate.npy")
PPO_train_completion_ratio_4.tolist()
PPO_train_completion_ratio_4=PPO_train_completion_ratio_4[:200]
PPO_train_ma_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_4.tolist()
PPO_train_ma_completion_ratio_4=PPO_train_ma_completion_ratio_4[:200]

PPO_train_delay_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_delay.npy")
PPO_train_delay_4.tolist()
PPO_train_delay=PPO_train_delay_4[:200]
PPO_train_ma_delay_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_delay.npy")
PPO_train_ma_delay_4.tolist()
PPO_train_ma_delay_4=PPO_train_ma_delay_4[:200]

PPO_train_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_energy_consumption.npy")
PPO_train_energy_consumption_4.tolist()
PPO_train_energy_consumption_4=PPO_train_energy_consumption_4[:200]
PPO_train_ma_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_4.tolist()
PPO_train_ma_energy_consumption_4=PPO_train_ma_energy_consumption_4[:200]
################################
# plot_PPO_rewards_lr(PPO_train_rewards_1,PPO_train_ma_rewards_1,PPO_train_rewards_2,PPO_train_ma_rewards_2,PPO_train_rewards_4,PPO_train_ma_rewards_4)
# plot_PPO_completion_ratio_lr(PPO_train_completion_ratio_1,PPO_train_ma_completion_ratio_1,PPO_train_completion_ratio_2,PPO_train_ma_completion_ratio_2,PPO_train_completion_ratio_4,PPO_train_ma_completion_ratio_4)
plot_PPO_delay_lr(PPO_train_delay_1,PPO_train_ma_delay_1,PPO_train_delay_2,PPO_train_ma_delay_2,PPO_train_delay_4,PPO_train_ma_delay_4)
plot_PPO_energy_consumption_lr(PPO_train_energy_consumption_1,PPO_train_ma_energy_consumption_1,PPO_train_energy_consumption_2,PPO_train_ma_energy_consumption_2,PPO_train_energy_consumption_4,PPO_train_ma_energy_consumption_4)
################################



#绘制绘制对比算法
PPO_train_delay = np.load("C:/Users/23928/Desktop/result/contrast/PPO/20240512-102413/results/train_delay.npy")
PPO_train_delay.tolist()
PPO_train_delay=PPO_train_delay[:200]
PPO_train_ma_delay = np.load("C:/Users/23928/Desktop/result/contrast/PPO/20240512-102413/results/train_ma_delay.npy")
PPO_train_ma_delay.tolist()
PPO_train_ma_delay=PPO_train_ma_delay[:200]


SAC_train_delay = np.load("C:/Users/23928/Desktop/result/contrast/SAC/20240510-192414/results/train_delay.npy")
SAC_train_delay.tolist()
SAC_train_delay=SAC_train_delay[:200]
SAC_train_ma_delay = np.load("C:/Users/23928/Desktop/result/contrast/SAC/20240510-192414/results/train_ma_delay.npy")
SAC_train_ma_delay.tolist()
SAC_train_ma_delay=SAC_train_ma_delay[:200]


Greedy_train_delay = np.load("C:/Users/23928/Desktop/result/contrast/Greedy/20240510-195712/results/train_delay.npy")
Greedy_train_delay.tolist()
Greedy_train_delay=Greedy_train_delay[:200]
Greedy_train_ma_delay = np.load("C:/Users/23928/Desktop/result/contrast/Greedy/20240510-195712/results/train_ma_delay.npy")
Greedy_train_ma_delay.tolist()
Greedy_train_ma_delay=Greedy_train_ma_delay[:200]
################################
plot_contrast_delay(PPO_train_delay,PPO_train_ma_delay,SAC_train_delay,SAC_train_ma_delay,Greedy_train_delay,Greedy_train_ma_delay)
################################





PPO_train_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/PPO/20240512-102413/results/train_completion_rate.npy")
PPO_train_completion_ratio.tolist()
PPO_train_completion_ratio=PPO_train_completion_ratio[:200]
PPO_train_ma_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/PPO/20240512-102413/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio.tolist()
PPO_train_ma_completion_ratio=PPO_train_ma_completion_ratio[:200]


SAC_train_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/SAC/20240510-192414/results/train_completion_rate.npy")
SAC_train_completion_ratio.tolist()
SAC_train_completion_ratio=SAC_train_completion_ratio[:200]
SAC_train_ma_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/SAC/20240510-192414/results/train_ma_completion_rate.npy")
SAC_train_ma_completion_ratio.tolist()
SAC_train_ma_completion_ratio=SAC_train_ma_completion_ratio[:200]


Greedy_train_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/Greedy/20240510-195712/results/train_completion_rate.npy")
Greedy_train_completion_ratio.tolist()
Greedy_train_completion_ratio=Greedy_train_completion_ratio[:200]
Greedy_train_ma_completion_ratio = np.load("C:/Users/23928/Desktop/result/contrast/Greedy/20240510-195712/results/train_ma_completion_rate.npy")
Greedy_train_ma_completion_ratio.tolist()
Greedy_train_ma_completion_ratio=Greedy_train_ma_completion_ratio[:200]
################################
# plot_contrast_completion_ratio(PPO_train_completion_ratio,PPO_train_ma_completion_ratio,SAC_train_completion_ratio,SAC_train_ma_completion_ratio,Greedy_train_completion_ratio,Greedy_train_ma_completion_ratio)
################################
#
#
# #绘制不同任务大小的收敛图
#
#
PPO_tsaksize_train_rewards_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_rewards.npy")
PPO_tsaksize_train_rewards_1.tolist()
PPO_tsaksize_train_rewards_1=PPO_tsaksize_train_rewards_1[:200]
PPO_tsaksize_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_ma_rewards.npy")
PPO_tsaksize_train_ma_rewards_1.tolist()
PPO_tsaksize_train_ma_rewards_1=PPO_tsaksize_train_ma_rewards_1[:200]

PPO_tsaksize_train_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_completion_rate.npy")
PPO_tsaksize_train_completion_ratio_1.tolist()
PPO_tsaksize_train_completion_ratio_1=PPO_tsaksize_train_completion_ratio_1[:200]
PPO_tsaksize_train_ma_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_ma_completion_rate.npy")
PPO_tsaksize_train_ma_completion_ratio_1.tolist()
PPO_tsaksize_train_ma_completion_ratio_1=PPO_tsaksize_train_ma_completion_ratio_1[:200]

PPO_tsaksize_train_delay_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_delay.npy")
PPO_tsaksize_train_delay_1.tolist()
PPO_tsaksize_train_delay_1=PPO_tsaksize_train_delay_1[:200]
# PPO_tsaksize_train_delay_1=sum(PPO_tsaksize_train_delay_1)/len(PPO_tsaksize_train_delay_1)
PPO_tsaksize_train_ma_delay_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_ma_delay.npy")
PPO_tsaksize_train_ma_delay_1.tolist()
PPO_tsaksize_train_ma_delay_1=PPO_tsaksize_train_ma_delay_1[:200]

PPO_tsaksize_train_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_energy_consumption.npy")
PPO_tsaksize_train_energy_consumption_1.tolist()
PPO_tsaksize_train_energy_consumption_1=PPO_tsaksize_train_energy_consumption_1[:200]
# PPO_tsaksize_train_energy_consumption_1=sum(PPO_tsaksize_train_energy_consumption_1)/len(PPO_tsaksize_train_energy_consumption_1)
PPO_tsaksize_train_ma_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/tasksize/2/PPO/20240429-102936/results/train_ma_energy_consumption.npy")
PPO_tsaksize_train_ma_energy_consumption_1.tolist()
PPO_tsaksize_train_ma_energy_consumption_1=PPO_tsaksize_train_ma_energy_consumption_1[:200]




PPO_tsaksize_train_rewards_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_rewards.npy")
PPO_tsaksize_train_rewards_2.tolist()
PPO_tsaksize_train_rewards_2=PPO_tsaksize_train_rewards_2[:200]
PPO_tsaksize_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_ma_rewards.npy")
PPO_tsaksize_train_ma_rewards_2.tolist()
PPO_tsaksize_train_ma_rewards_2=PPO_tsaksize_train_ma_rewards_2[:200]

PPO_tsaksize_train_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_completion_rate.npy")
PPO_tsaksize_train_completion_ratio_2.tolist()
PPO_tsaksize_train_completion_ratio_2=PPO_tsaksize_train_completion_ratio_2[:200]
PPO_tsaksize_train_ma_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_ma_completion_rate.npy")
PPO_tsaksize_train_ma_completion_ratio_2.tolist()
PPO_tsaksize_train_ma_completion_ratio_2=PPO_tsaksize_train_ma_completion_ratio_2[:200]

PPO_tsaksize_train_delay_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_delay.npy")
PPO_tsaksize_train_delay_2.tolist()
PPO_tsaksize_train_delay_2=PPO_tsaksize_train_delay_2[:200]
# PPO_tsaksize_train_delay_2=sum(PPO_tsaksize_train_delay_2)/len(PPO_tsaksize_train_delay_2)
PPO_tsaksize_train_ma_delay_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_ma_delay.npy")
PPO_tsaksize_train_ma_delay_2.tolist()
PPO_tsaksize_train_ma_delay_2=PPO_tsaksize_train_ma_delay_2[:200]

PPO_tsaksize_train_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_energy_consumption.npy")
PPO_tsaksize_train_energy_consumption_2.tolist()
PPO_tsaksize_train_energy_consumption_2=PPO_tsaksize_train_energy_consumption_2[:200]
# PPO_tsaksize_train_energy_consumption_2=sum(PPO_tsaksize_train_energy_consumption_2)/len(PPO_tsaksize_train_energy_consumption_2)
PPO_tsaksize_train_ma_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/tasksize/3/PPO/20240429-092134/results/train_ma_energy_consumption.npy")
PPO_tsaksize_train_ma_energy_consumption_2.tolist()
PPO_tsaksize_train_ma_energy_consumption_2=PPO_tsaksize_train_ma_energy_consumption_2[:200]





PPO_tsaksize_train_rewards_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_rewards.npy")
PPO_tsaksize_train_rewards_3.tolist()
PPO_tsaksize_train_rewards_3=PPO_tsaksize_train_rewards_3[:200]
PPO_tsaksize_train_ma_rewards_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_ma_rewards.npy")
PPO_tsaksize_train_ma_rewards_3.tolist()
PPO_tsaksize_train_ma_rewards_3=PPO_tsaksize_train_ma_rewards_3[:200]

PPO_tsaksize_train_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_completion_rate.npy")
PPO_tsaksize_train_completion_ratio_3.tolist()
PPO_tsaksize_train_completion_ratio_3=PPO_tsaksize_train_completion_ratio_3[:200]
PPO_tsaksize_train_ma_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_ma_completion_rate.npy")
PPO_tsaksize_train_ma_completion_ratio_3.tolist()
PPO_tsaksize_train_ma_completion_ratio_3=PPO_tsaksize_train_ma_completion_ratio_3[:200]

PPO_tsaksize_train_delay_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_delay.npy")
PPO_tsaksize_train_delay_3.tolist()
PPO_tsaksize_train_delay_3=PPO_tsaksize_train_delay_3[:200]
# PPO_tsaksize_train_delay_3=sum(PPO_tsaksize_train_delay_3)/len(PPO_tsaksize_train_delay_3)
PPO_tsaksize_train_ma_delay_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_ma_delay.npy")
PPO_tsaksize_train_ma_delay_3.tolist()
PPO_tsaksize_train_ma_delay_3=PPO_tsaksize_train_ma_delay_3[:200]

PPO_tsaksize_train_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_energy_consumption.npy")
PPO_tsaksize_train_energy_consumption_3.tolist()
PPO_tsaksize_train_energy_consumption_3=PPO_tsaksize_train_energy_consumption_3[:200]
# PPO_tsaksize_train_energy_consumption_3=sum(PPO_tsaksize_train_energy_consumption_3)/len(PPO_tsaksize_train_energy_consumption_3)
PPO_tsaksize_train_ma_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/tasksize/4/PPO/20240429-201651/results/train_ma_energy_consumption.npy")
PPO_tsaksize_train_ma_energy_consumption_3.tolist()
PPO_tsaksize_train_ma_energy_consumption_3=PPO_tsaksize_train_ma_energy_consumption_3[:200]




PPO_tsaksize_train_rewards_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_rewards.npy")
PPO_tsaksize_train_rewards_4.tolist()
PPO_tsaksize_train_rewards_4=PPO_tsaksize_train_rewards_4[50:200]
PPO_tsaksize_train_ma_rewards_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_ma_rewards.npy")
PPO_tsaksize_train_ma_rewards_4.tolist()
PPO_tsaksize_train_ma_rewards_4=PPO_tsaksize_train_ma_rewards_4[50:200]

PPO_tsaksize_train_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_completion_rate.npy")
PPO_tsaksize_train_completion_ratio_4.tolist()
PPO_tsaksize_train_completion_ratio_4=PPO_tsaksize_train_completion_ratio_4[:200]
PPO_tsaksize_train_ma_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_ma_completion_rate.npy")
PPO_tsaksize_train_ma_completion_ratio_4.tolist()
PPO_tsaksize_train_ma_completion_ratio_4=PPO_tsaksize_train_ma_completion_ratio_4[:200]

PPO_tsaksize_train_delay_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_delay.npy")
PPO_tsaksize_train_delay_4.tolist()
PPO_tsaksize_train_delay_4=PPO_tsaksize_train_delay_4[:200]
# PPO_tsaksize_train_delay_3=sum(PPO_tsaksize_train_delay_3)/len(PPO_tsaksize_train_delay_3)
PPO_tsaksize_train_ma_delay_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_ma_delay.npy")
PPO_tsaksize_train_ma_delay_4.tolist()
PPO_tsaksize_train_ma_delay_4=PPO_tsaksize_train_ma_delay_4[:200]

PPO_tsaksize_train_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_energy_consumption.npy")
PPO_tsaksize_train_energy_consumption_4.tolist()
PPO_tsaksize_train_energy_consumption_4=PPO_tsaksize_train_energy_consumption_4[:200]
# PPO_tsaksize_train_energy_consumption_3=sum(PPO_tsaksize_train_energy_consumption_3)/len(PPO_tsaksize_train_energy_consumption_3)
PPO_tsaksize_train_ma_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/tasksize/5/PPO/20240508-180753/results/train_ma_energy_consumption.npy")
PPO_tsaksize_train_ma_energy_consumption_4.tolist()
PPO_tsaksize_train_ma_energy_consumption_4=PPO_tsaksize_train_ma_energy_consumption_4[:200]

################################
# plot_tasksize_rewards_box_diagram(PPO_tsaksize_train_rewards_1,PPO_tsaksize_train_rewards_2,PPO_tsaksize_train_rewards_3)
# plot_tasksize_completion_ratio_box_diagram(PPO_tsaksize_train_completion_ratio_1,PPO_tsaksize_train_completion_ratio_2,PPO_tsaksize_train_completion_ratio_3)
plot_tasksize_delay_box_diagram(PPO_tsaksize_train_delay_1,PPO_tsaksize_train_delay_2,PPO_tsaksize_train_delay_3)
plot_tasksize_energy_consumption_box_diagram(PPO_tsaksize_train_energy_consumption_1,PPO_tsaksize_train_energy_consumption_2,PPO_tsaksize_train_energy_consumption_3)


# plot_tasksize_rewards(PPO_tsaksize_train_rewards_1,PPO_tsaksize_train_ma_rewards_1,PPO_tsaksize_train_rewards_2,PPO_tsaksize_train_ma_rewards_2,PPO_tsaksize_train_rewards_3,PPO_tsaksize_train_ma_rewards_3)
# plot_tasksize_completion_ratio(PPO_tsaksize_train_completion_ratio_1,PPO_tsaksize_train_ma_completion_ratio_1,PPO_tsaksize_train_completion_ratio_2,PPO_tsaksize_train_ma_completion_ratio_2,PPO_tsaksize_train_completion_ratio_3,PPO_tsaksize_train_ma_completion_ratio_3)
# plot_tasksize_delay(PPO_tsaksize_train_delay_1,PPO_tsaksize_train_ma_delay_1,PPO_tsaksize_train_delay_2,PPO_tsaksize_train_ma_delay_2,PPO_tsaksize_train_delay_3,PPO_tsaksize_train_ma_delay_3)
# plot_tasksize_energy_consumption(PPO_tsaksize_train_energy_consumption_1,PPO_tsaksize_train_ma_energy_consumption_1,PPO_tsaksize_train_energy_consumption_2,PPO_tsaksize_train_ma_energy_consumption_2,PPO_tsaksize_train_energy_consumption_3,PPO_tsaksize_train_ma_energy_consumption_3)
################################



# plot_tasksize_four_dimensional(PPO_tsaksize_train_rewards_1,PPO_tsaksize_train_delay_1,PPO_tsaksize_train_energy_consumption_1,PPO_tsaksize_train_rewards_2,PPO_tsaksize_train_delay_2,PPO_tsaksize_train_energy_consumption_2,PPO_tsaksize_train_rewards_3,PPO_tsaksize_train_delay_3,PPO_tsaksize_train_energy_consumption_3)
# plot_tasksize_delay_line_chart(PPO_tsaksize_train_delay_1,PPO_tsaksize_train_delay_2,PPO_tsaksize_train_delay_3)
# plot_tasksize_energy_consumption_line_chart(PPO_tsaksize_train_energy_consumption_1,PPO_tsaksize_train_energy_consumption_2,PPO_tsaksize_train_energy_consumption_3)
# plot_box_diagram(PPO_tsaksize_train_rewards_1, PPO_tsaksize_train_rewards_2, PPO_tsaksize_train_rewards_3, labels=['tasksize=2', 'tasksize=3', 'tasksize=4'])
# plot_box_diagram(PPO_tsaksize_train_delay_1, PPO_tsaksize_train_delay_2, PPO_tsaksize_train_delay_3, labels=['tasksize=2', 'tasksize=3', 'tasksize=4'])
# plot_box_diagram(PPO_tsaksize_train_energy_consumption_1, PPO_tsaksize_train_energy_consumption_2, PPO_tsaksize_train_energy_consumption_3, labels=['tasksize=2', 'tasksize=3', 'tasksize=4'])
# #绘制不同计算资源的收敛图

PPO_computing_resource_train_rewards_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_rewards.npy")
PPO_computing_resource_train_rewards_1.tolist()
PPO_computing_resource_train_rewards_1=PPO_computing_resource_train_rewards_1[:200]
PPO_computing_resource_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_rewards.npy")
PPO_computing_resource_train_ma_rewards_1.tolist()
PPO_computing_resource_train_ma_rewards_1=PPO_computing_resource_train_ma_rewards_1[:200]

PPO_computing_resource_train_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_completion_rate.npy")
PPO_computing_resource_train_completion_ratio_1.tolist()
PPO_computing_resource_train_completion_ratio_1=PPO_computing_resource_train_completion_ratio_1[:200]
PPO_computing_resource_train_ma_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_completion_rate.npy")
PPO_computing_resource_train_ma_completion_ratio_1.tolist()
PPO_computing_resource_train_ma_completion_ratio_1=PPO_computing_resource_train_ma_completion_ratio_1[:200]

PPO_computing_resource_train_delay_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_delay.npy")
PPO_computing_resource_train_delay_1.tolist()
PPO_computing_resource_train_delay_1=PPO_computing_resource_train_delay_1[:200]
PPO_computing_resource_train_ma_delay_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_delay.npy")
PPO_computing_resource_train_ma_delay_1.tolist()
PPO_computing_resource_train_ma_delay_1=PPO_computing_resource_train_ma_delay_1[:200]

PPO_computing_resource_train_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_energy_consumption.npy")
PPO_computing_resource_train_energy_consumption_1.tolist()
PPO_computing_resource_train_energy_consumption_1=PPO_computing_resource_train_energy_consumption_1[:200]
PPO_computing_resource_train_ma_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_computing_resource_train_ma_energy_consumption_1.tolist()
PPO_computing_resource_train_ma_energy_consumption_1=PPO_computing_resource_train_ma_energy_consumption_1[:200]


#
PPO_computing_resource_train_rewards_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_rewards.npy")
PPO_computing_resource_train_rewards_2.tolist()
PPO_computing_resource_train_rewards_2=PPO_computing_resource_train_rewards_2[:200]
PPO_computing_resource_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_ma_rewards.npy")
PPO_computing_resource_train_ma_rewards_2.tolist()
PPO_computing_resource_train_ma_rewards_2=PPO_computing_resource_train_ma_rewards_2[:200]

PPO_computing_resource_train_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_completion_rate.npy")
PPO_computing_resource_train_completion_ratio_2.tolist()
PPO_computing_resource_train_completion_ratio_2=PPO_computing_resource_train_completion_ratio_2[:200]
PPO_computing_resource_train_ma_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_ma_completion_rate.npy")
PPO_computing_resource_train_ma_completion_ratio_2.tolist()
PPO_computing_resource_train_ma_completion_ratio_2=PPO_computing_resource_train_ma_completion_ratio_2[:200]

PPO_computing_resource_train_delay_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_delay.npy")
PPO_computing_resource_train_delay_2.tolist()
PPO_computing_resource_train_delay_2=PPO_computing_resource_train_delay_2[:200]
PPO_computing_resource_train_ma_delay_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_ma_delay.npy")
PPO_computing_resource_train_ma_delay_2.tolist()
PPO_computing_resource_train_ma_delay_2=PPO_computing_resource_train_ma_delay_2[:200]

PPO_computing_resource_train_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_energy_consumption.npy")
PPO_computing_resource_train_energy_consumption_2.tolist()
PPO_computing_resource_train_energy_consumption_2=PPO_computing_resource_train_energy_consumption_2[:200]
PPO_computing_resource_train_ma_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/compute resource/325/PPO/20240506-201741/results/train_ma_energy_consumption.npy")
PPO_computing_resource_train_ma_energy_consumption_2.tolist()
PPO_computing_resource_train_ma_energy_consumption_2=PPO_computing_resource_train_ma_energy_consumption_2[:200]
#
#
PPO_computing_resource_train_rewards_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_rewards.npy")
PPO_computing_resource_train_rewards_3.tolist()
PPO_computing_resource_train_rewards_3=PPO_computing_resource_train_rewards_3[:200]
PPO_computing_resource_train_ma_rewards_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_ma_rewards.npy")
PPO_computing_resource_train_ma_rewards_3.tolist()
PPO_computing_resource_train_ma_rewards_3=PPO_computing_resource_train_ma_rewards_3[:200]

PPO_computing_resource_train_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_completion_rate.npy")
PPO_computing_resource_train_completion_ratio_3.tolist()
PPO_computing_resource_train_completion_ratio_3=PPO_computing_resource_train_completion_ratio_3[:200]
PPO_computing_resource_train_ma_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_ma_completion_rate.npy")
PPO_computing_resource_train_ma_completion_ratio_3.tolist()
PPO_computing_resource_train_ma_completion_ratio_3=PPO_computing_resource_train_ma_completion_ratio_3[:200]

PPO_computing_resource_train_delay_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_delay.npy")
PPO_computing_resource_train_delay_3.tolist()
PPO_computing_resource_train_delay_3=PPO_computing_resource_train_delay_3[:200]
PPO_computing_resource_train_ma_delay_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_ma_delay.npy")
PPO_computing_resource_train_ma_delay_3.tolist()
PPO_computing_resource_train_ma_delay_3=PPO_computing_resource_train_ma_delay_3[:200]

PPO_computing_resource_train_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_energy_consumption.npy")
PPO_computing_resource_train_energy_consumption_3.tolist()
PPO_computing_resource_train_energy_consumption_3=PPO_computing_resource_train_energy_consumption_3[:200]
PPO_computing_resource_train_ma_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/compute resource/350/PPO/20240506-215728/results/train_ma_energy_consumption.npy")
PPO_computing_resource_train_ma_energy_consumption_3.tolist()
PPO_computing_resource_train_ma_energy_consumption_3=PPO_computing_resource_train_ma_energy_consumption_3[:200]
################################
# plot_computing_resource_rewards_box_diagram(PPO_computing_resource_train_rewards_1,PPO_computing_resource_train_rewards_2,PPO_computing_resource_train_rewards_3)
# plot_computing_resource_completion_ratio_box_diagram(PPO_computing_resource_train_completion_ratio_1,PPO_computing_resource_train_completion_ratio_2,PPO_computing_resource_train_completion_ratio_3)
# plot_computing_resource_delay_box_diagram(PPO_computing_resource_train_delay_1,PPO_computing_resource_train_delay_2,PPO_computing_resource_train_delay_3)
# plot_computing_resource_energy_consumption_box_diagram(PPO_computing_resource_train_energy_consumption_1,PPO_computing_resource_train_energy_consumption_2,PPO_computing_resource_train_energy_consumption_3)

# plot_computing_resource_rewards(PPO_computing_resource_train_ma_rewards_1,PPO_computing_resource_train_ma_rewards_2,PPO_computing_resource_train_ma_rewards_3)
# plot_computing_resource_completion_ratio(PPO_computing_resource_train_ma_completion_ratio_1,PPO_computing_resource_train_ma_completion_ratio_2,PPO_computing_resource_train_ma_completion_ratio_3)
# plot_computing_resource_delay(PPO_computing_resource_train_ma_delay_1,PPO_computing_resource_train_ma_delay_2,PPO_computing_resource_train_ma_delay_3)
# plot_computing_resource_energy_consumption(PPO_computing_resource_train_ma_energy_consumption_1,PPO_computing_resource_train_ma_energy_consumption_2,PPO_computing_resource_train_ma_energy_consumption_3)
################################
#
#绘制不同车辆数目的收敛图
PPO_vehicle_number_train_rewards_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_rewards.npy")
PPO_vehicle_number_train_rewards_6.tolist()
PPO_vehicle_number_train_rewards_6=PPO_vehicle_number_train_rewards_6[:200]
PPO_vehicle_number_train_ma_rewards_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_ma_rewards.npy")
PPO_vehicle_number_train_ma_rewards_6.tolist()
PPO_vehicle_number_train_ma_rewards_6=PPO_vehicle_number_train_ma_rewards_6[:200]

PPO_vehicle_number_train_completion_ratio_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_completion_rate.npy")
PPO_vehicle_number_train_completion_ratio_6.tolist()
PPO_vehicle_number_train_completion_ratio_6=PPO_vehicle_number_train_completion_ratio_6[:200]
PPO_vehicle_number_train_ma_completion_ratio_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_ma_completion_rate.npy")
PPO_vehicle_number_train_ma_completion_ratio_6.tolist()
PPO_vehicle_number_train_ma_completion_ratio_6=PPO_vehicle_number_train_ma_completion_ratio_6[:200]

PPO_vehicle_number_train_delay_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_delay.npy")
PPO_vehicle_number_train_delay_6.tolist()
PPO_vehicle_number_train_delay_6=PPO_vehicle_number_train_delay_6[:200]
PPO_vehicle_number_train_ma_delay_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_ma_delay.npy")
PPO_vehicle_number_train_ma_delay_6.tolist()
PPO_vehicle_number_train_ma_delay_6=PPO_vehicle_number_train_ma_delay_6[:200]

PPO_vehicle_number_train_energy_consumption_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_energy_consumption.npy")
PPO_vehicle_number_train_energy_consumption_6.tolist()
PPO_vehicle_number_train_energy_consumption_6=PPO_vehicle_number_train_energy_consumption_6[:200]
PPO_vehicle_number_train_ma_energy_consumption_6 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/6/20240503-112248/results/train_ma_energy_consumption.npy")
PPO_vehicle_number_train_ma_energy_consumption_6.tolist()
PPO_vehicle_number_train_ma_energy_consumption_6=PPO_vehicle_number_train_ma_energy_consumption_6[:200]





PPO_vehicle_number_train_rewards_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_rewards.npy")
PPO_vehicle_number_train_rewards_8.tolist()
PPO_vehicle_number_train_rewards_8=PPO_vehicle_number_train_rewards_8[:200]
PPO_vehicle_number_train_ma_rewards_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_ma_rewards.npy")
PPO_vehicle_number_train_ma_rewards_8.tolist()
PPO_vehicle_number_train_ma_rewards_8=PPO_vehicle_number_train_ma_rewards_8[:200]

PPO_vehicle_number_train_completion_ratio_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_completion_rate.npy")
PPO_vehicle_number_train_completion_ratio_8.tolist()
PPO_vehicle_number_train_completion_ratio_8=PPO_vehicle_number_train_completion_ratio_8[:200]
PPO_vehicle_number_train_ma_completion_ratio_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_ma_completion_rate.npy")
PPO_vehicle_number_train_ma_completion_ratio_8.tolist()
PPO_vehicle_number_train_ma_completion_ratio_8=PPO_vehicle_number_train_ma_completion_ratio_8[:200]

PPO_vehicle_number_train_delay_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_delay.npy")
PPO_vehicle_number_train_delay_8.tolist()
PPO_vehicle_number_train_delay_8=PPO_vehicle_number_train_delay_8[:200]
PPO_vehicle_number_train_ma_delay_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_ma_delay.npy")
PPO_vehicle_number_train_ma_delay_8.tolist()
PPO_vehicle_number_train_ma_delay_8=PPO_vehicle_number_train_ma_delay_8[:200]

PPO_vehicle_number_train_energy_consumption_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_energy_consumption.npy")
PPO_vehicle_number_train_energy_consumption_8.tolist()
PPO_vehicle_number_train_energy_consumption_8=PPO_vehicle_number_train_energy_consumption_8[:200]
PPO_vehicle_number_train_ma_energy_consumption_8 = np.load("C:/Users/23928/Desktop/result/vehicle number/PPO/8/20240504-182114/results/train_ma_energy_consumption.npy")
PPO_vehicle_number_train_ma_energy_consumption_8.tolist()
PPO_vehicle_number_train_ma_energy_consumption_8=PPO_vehicle_number_train_ma_energy_consumption_8[:200]




PPO_vehicle_number_train_rewards_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_rewards.npy")
PPO_vehicle_number_train_rewards_10.tolist()
PPO_vehicle_number_train_rewards_10=PPO_vehicle_number_train_rewards_10[:200]
PPO_vehicle_number_train_ma_rewards_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_rewards.npy")
PPO_vehicle_number_train_ma_rewards_10.tolist()
PPO_vehicle_number_train_ma_rewards_10=PPO_vehicle_number_train_ma_rewards_10[:200]

PPO_vehicle_number_train_completion_ratio_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_completion_rate.npy")
PPO_vehicle_number_train_completion_ratio_10.tolist()
PPO_vehicle_number_train_completion_ratio_10=PPO_vehicle_number_train_completion_ratio_10[:200]
PPO_vehicle_number_train_ma_completion_ratio_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_completion_rate.npy")
PPO_vehicle_number_train_ma_completion_ratio_10.tolist()
PPO_vehicle_number_train_ma_completion_ratio_10=PPO_vehicle_number_train_ma_completion_ratio_10[:200]

PPO_vehicle_number_train_delay_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_delay.npy")
PPO_vehicle_number_train_delay_10.tolist()
PPO_vehicle_number_train_delay_10=PPO_vehicle_number_train_delay_10[:200]
PPO_vehicle_number_train_ma_delay_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_delay.npy")
PPO_vehicle_number_train_ma_delay_10.tolist()
PPO_vehicle_number_train_ma_delay_10=PPO_vehicle_number_train_ma_delay_10[:200]

PPO_vehicle_number_train_energy_consumption_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_energy_consumption.npy")
PPO_vehicle_number_train_energy_consumption_10.tolist()
PPO_vehicle_number_train_energy_consumption_10=PPO_vehicle_number_train_energy_consumption_10[:200]
PPO_vehicle_number_train_ma_energy_consumption_10 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_vehicle_number_train_ma_energy_consumption_10.tolist()
PPO_vehicle_number_train_ma_energy_consumption_10=PPO_vehicle_number_train_ma_energy_consumption_10[:200]
# plot_box_diagram(PPO_vehicle_number_train_rewards_6, PPO_vehicle_number_train_rewards_8, PPO_vehicle_number_train_rewards_10, labels=['vechile number=2', 'vechile number=3', 'vechile number=4'])
################################
# plot_vehicle_number_delay_box_diagram(PPO_vehicle_number_train_delay_6,PPO_vehicle_number_train_delay_8,PPO_vehicle_number_train_delay_10)
# plot_vehicle_number_energy_consumption_box_diagram(PPO_vehicle_number_train_energy_consumption_6,PPO_vehicle_number_train_energy_consumption_8,PPO_vehicle_number_train_energy_consumption_10)

# plot_vehicle_number_completion_ratio_box_diagram(PPO_vehicle_number_train_completion_ratio_6,PPO_vehicle_number_train_completion_ratio_8,PPO_vehicle_number_train_completion_ratio_10)
#
plot_vehicle_number_delay_and_energy_consumption_bar_chart(PPO_vehicle_number_train_delay_6,PPO_vehicle_number_train_delay_8,PPO_vehicle_number_train_delay_10,PPO_vehicle_number_train_energy_consumption_6,PPO_vehicle_number_train_energy_consumption_8,PPO_vehicle_number_train_energy_consumption_10)
#
# plot_vehicle_number_rewards(PPO_vehicle_number_train_rewards_6,PPO_vehicle_number_train_ma_rewards_6,PPO_vehicle_number_train_rewards_8,PPO_vehicle_number_train_ma_rewards_8,PPO_vehicle_number_train_rewards_10,PPO_vehicle_number_train_ma_rewards_10)
# plot_vehicle_number_completion_ratio(PPO_vehicle_number_train_ma_completion_ratio_6,PPO_vehicle_number_train_ma_completion_ratio_8,PPO_vehicle_number_train_ma_completion_ratio_10)
# plot_vehicle_number_delay(PPO_vehicle_number_train_ma_delay_6,PPO_vehicle_number_train_ma_delay_8,PPO_vehicle_number_train_ma_delay_10)
# plot_vehicle_number_energy_consumption(PPO_vehicle_number_train_ma_energy_consumption_6,PPO_vehicle_number_train_ma_energy_consumption_8,PPO_vehicle_number_train_ma_energy_consumption_10)
################################

#
#
#
#
# 绘制不同车辆通信范围的收敛图
PPO_vehicle_range_train_rewards_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_rewards.npy")
PPO_vehicle_range_train_rewards_100.tolist()
PPO_vehicle_range_train_rewards_100=PPO_vehicle_range_train_rewards_100[:200]
PPO_vehicle_range_train_ma_rewards_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_ma_rewards.npy")
PPO_vehicle_range_train_ma_rewards_100.tolist()
PPO_vehicle_range_train_ma_rewards_100=PPO_vehicle_range_train_ma_rewards_100[:200]

PPO_vehicle_range_train_completion_ratio_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_completion_rate.npy")
PPO_vehicle_range_train_completion_ratio_100.tolist()
PPO_vehicle_range_train_completion_ratio_100=PPO_vehicle_range_train_completion_ratio_100[:200]
PPO_vehicle_range_train_ma_completion_ratio_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_ma_completion_rate.npy")
PPO_vehicle_range_train_ma_completion_ratio_100.tolist()
PPO_vehicle_range_train_ma_completion_ratio_100=PPO_vehicle_range_train_ma_completion_ratio_100[:200]

PPO_vehicle_range_train_delay_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_delay.npy")
PPO_vehicle_range_train_delay_100.tolist()
PPO_vehicle_range_train_delay_100=PPO_vehicle_range_train_delay_100[:200]
PPO_vehicle_range_train_ma_delay_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_ma_delay.npy")
PPO_vehicle_range_train_ma_delay_100.tolist()
PPO_vehicle_range_train_ma_delay_100=PPO_vehicle_range_train_ma_delay_100[:200]

PPO_vehicle_range_train_energy_consumption_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_energy_consumption.npy")
PPO_vehicle_range_train_energy_consumption_100.tolist()
PPO_vehicle_range_train_energy_consumption_100=PPO_vehicle_range_train_energy_consumption_100[:200]
PPO_vehicle_range_train_ma_energy_consumption_100 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/100/20240506-214303/results/train_ma_energy_consumption.npy")
PPO_vehicle_range_train_ma_energy_consumption_100.tolist()
PPO_vehicle_range_train_ma_energy_consumption_100=PPO_vehicle_range_train_ma_energy_consumption_100[:200]



PPO_vehicle_range_train_rewards_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_rewards.npy")
PPO_vehicle_range_train_rewards_125.tolist()
PPO_vehicle_range_train_rewards_125=PPO_vehicle_range_train_rewards_125[:200]
PPO_vehicle_range_train_ma_rewards_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_ma_rewards.npy")
PPO_vehicle_range_train_ma_rewards_125.tolist()
PPO_vehicle_range_train_ma_rewards_125=PPO_vehicle_range_train_ma_rewards_125[:200]

PPO_vehicle_range_train_completion_ratio_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_completion_rate.npy")
PPO_vehicle_range_train_completion_ratio_125.tolist()
PPO_vehicle_range_train_completion_ratio_125=PPO_vehicle_range_train_completion_ratio_125[:200]
PPO_vehicle_range_train_ma_completion_ratio_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_ma_completion_rate.npy")
PPO_vehicle_range_train_ma_completion_ratio_125.tolist()
PPO_vehicle_range_train_ma_completion_ratio_125=PPO_vehicle_range_train_ma_completion_ratio_125[:200]

PPO_vehicle_range_train_delay_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_delay.npy")
PPO_vehicle_range_train_delay_125.tolist()
PPO_vehicle_range_train_delay_125=PPO_vehicle_range_train_delay_125[:200]
PPO_vehicle_range_train_ma_delay_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_ma_delay.npy")
PPO_vehicle_range_train_ma_delay_125.tolist()
PPO_vehicle_range_train_ma_delay_125=PPO_vehicle_range_train_ma_delay_125[:200]

PPO_vehicle_range_train_energy_consumption_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_energy_consumption.npy")
PPO_vehicle_range_train_energy_consumption_125.tolist()
PPO_vehicle_range_train_energy_consumption_125=PPO_vehicle_range_train_energy_consumption_125[:200]
PPO_vehicle_range_train_ma_energy_consumption_125 = np.load("C:/Users/23928/Desktop/result/vehicle range/PPO/125/20240502-154645/results/train_ma_energy_consumption.npy")
PPO_vehicle_range_train_ma_energy_consumption_125.tolist()
PPO_vehicle_range_train_ma_energy_consumption_125=PPO_vehicle_range_train_ma_energy_consumption_125[:200]




PPO_vehicle_range_train_rewards_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_rewards.npy")
PPO_vehicle_range_train_rewards_150.tolist()
PPO_vehicle_range_train_rewards_150=PPO_vehicle_range_train_rewards_150[:200]
PPO_vehicle_range_train_ma_rewards_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_rewards.npy")
PPO_vehicle_range_train_ma_rewards_150.tolist()
PPO_vehicle_range_train_ma_rewards_150=PPO_vehicle_range_train_ma_rewards_150[:200]

PPO_vehicle_range_train_completion_ratio_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_completion_rate.npy")
PPO_vehicle_range_train_completion_ratio_150.tolist()
PPO_vehicle_range_train_completion_ratio_150=PPO_vehicle_range_train_completion_ratio_150[:200]
PPO_vehicle_range_train_ma_completion_ratio_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_completion_rate.npy")
PPO_vehicle_range_train_ma_completion_ratio_150.tolist()
PPO_vehicle_range_train_ma_completion_ratio_150=PPO_vehicle_range_train_ma_completion_ratio_150[:200]

PPO_vehicle_range_train_delay_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_delay.npy")
PPO_vehicle_range_train_delay_150.tolist()
PPO_vehicle_range_train_delay_150=PPO_vehicle_range_train_delay_150[:200]
PPO_vehicle_range_train_ma_delay_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_delay.npy")
PPO_vehicle_range_train_ma_delay_150.tolist()
PPO_vehicle_range_train_ma_delay_150=PPO_vehicle_range_train_ma_delay_150[:200]


PPO_vehicle_range_train_energy_consumption_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_energy_consumption.npy")
PPO_vehicle_range_train_energy_consumption_150.tolist()
PPO_vehicle_range_train_energy_consumption_150=PPO_vehicle_range_train_energy_consumption_150[:200]
PPO_vehicle_range_train_ma_energy_consumption_150 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_vehicle_range_train_ma_energy_consumption_150.tolist()
PPO_vehicle_range_train_ma_energy_consumption_150=PPO_vehicle_range_train_ma_energy_consumption_150[:200]

################################
# plot_vehicle_range_completion_ratio_box_diagram(PPO_vehicle_range_train_completion_ratio_100,PPO_vehicle_range_train_completion_ratio_125,PPO_vehicle_range_train_completion_ratio_150)
#
plot_vehicle_range_delay_and_energy_consumption_bar_chart(PPO_vehicle_range_train_delay_100,PPO_vehicle_range_train_delay_125,PPO_vehicle_range_train_delay_150,PPO_vehicle_range_train_energy_consumption_100,PPO_vehicle_range_train_energy_consumption_125,PPO_vehicle_range_train_energy_consumption_150)
#
# plot_vehicle_range_rewards(PPO_vehicle_range_train_rewards_100,PPO_vehicle_range_train_ma_rewards_100,PPO_vehicle_range_train_rewards_125,PPO_vehicle_range_train_ma_rewards_125,PPO_vehicle_range_train_rewards_150,PPO_vehicle_range_train_ma_rewards_150)
# plot_vehicle_range_completion_ratio(PPO_vehicle_range_train_ma_completion_ratio_100,PPO_vehicle_range_train_ma_completion_ratio_125,PPO_vehicle_range_train_ma_completion_ratio_150)
# plot_vehicle_range_delay(PPO_vehicle_range_train_ma_delay_100,PPO_vehicle_range_train_ma_delay_125,PPO_vehicle_range_train_ma_delay_150)
# plot_vehicle_range_energy_consumption(PPO_vehicle_range_train_ma_energy_consumption_100,PPO_vehicle_range_train_ma_energy_consumption_125,PPO_vehicle_range_train_ma_energy_consumption_150)
################################
#
# # #
#绘制不同车辆速度的收敛图
PPO_vehicle_speed_train_rewards_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_rewards.npy")
PPO_vehicle_speed_train_rewards_1.tolist()
PPO_vehicle_speed_train_rewards_1=PPO_vehicle_speed_train_rewards_1[:200]
PPO_vehicle_speed_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_ma_rewards.npy")
PPO_vehicle_speed_train_ma_rewards_1.tolist()
PPO_vehicle_speed_train_ma_rewards_1=PPO_vehicle_speed_train_ma_rewards_1[:200]

PPO_vehicle_speed_train_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_completion_rate.npy")
PPO_vehicle_speed_train_completion_ratio_1.tolist()
PPO_vehicle_speed_train_completion_ratio_1=PPO_vehicle_speed_train_completion_ratio_1[:200]
PPO_vehicle_speed_train_ma_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_ma_completion_rate.npy")
PPO_vehicle_speed_train_ma_completion_ratio_1.tolist()
PPO_vehicle_speed_train_ma_completion_ratio_1=PPO_vehicle_speed_train_ma_completion_ratio_1[:200]

PPO_vehicle_speed_train_delay_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_delay.npy")
PPO_vehicle_speed_train_delay_1.tolist()
PPO_vehicle_speed_train_delay_1=PPO_vehicle_speed_train_delay_1[:200]
PPO_vehicle_speed_train_ma_delay_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_ma_delay.npy")
PPO_vehicle_speed_train_ma_delay_1.tolist()
PPO_vehicle_speed_train_ma_delay_1=PPO_vehicle_speed_train_ma_delay_1[:200]

PPO_vehicle_speed_train_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_energy_consumption.npy")
PPO_vehicle_speed_train_energy_consumption_1.tolist()
PPO_vehicle_speed_train_energy_consumption_1=PPO_vehicle_speed_train_energy_consumption_1[:200]
PPO_vehicle_speed_train_ma_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/20-30/20240506-214251/results/train_ma_energy_consumption.npy")
PPO_vehicle_speed_train_ma_energy_consumption_1.tolist()
PPO_vehicle_speed_train_ma_energy_consumption_1=PPO_vehicle_speed_train_ma_energy_consumption_1[:200]




PPO_vehicle_speed_train_rewards_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_rewards.npy")
PPO_vehicle_speed_train_rewards_2.tolist()
PPO_vehicle_speed_train_rewards_2=PPO_vehicle_speed_train_rewards_2[:200]
PPO_vehicle_speed_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_ma_rewards.npy")
PPO_vehicle_speed_train_ma_rewards_2.tolist()
PPO_vehicle_speed_train_ma_rewards_2=PPO_vehicle_speed_train_ma_rewards_2[:200]

PPO_vehicle_speed_train_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_completion_rate.npy")
PPO_vehicle_speed_train_completion_ratio_2.tolist()
PPO_vehicle_speed_train_completion_ratio_2=PPO_vehicle_speed_train_completion_ratio_2[:200]
PPO_vehicle_speed_train_ma_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_ma_completion_rate.npy")
PPO_vehicle_speed_train_ma_completion_ratio_2.tolist()
PPO_vehicle_speed_train_ma_completion_ratio_2=PPO_vehicle_speed_train_ma_completion_ratio_2[:200]

PPO_vehicle_speed_train_delay_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_delay.npy")
PPO_vehicle_speed_train_delay_2.tolist()
PPO_vehicle_speed_train_delay_2=PPO_vehicle_speed_train_delay_2[:200]
PPO_vehicle_speed_train_ma_delay_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_ma_delay.npy")
PPO_vehicle_speed_train_ma_delay_2.tolist()
PPO_vehicle_speed_train_ma_delay_2=PPO_vehicle_speed_train_ma_delay_2[:200]

PPO_vehicle_speed_train_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_energy_consumption.npy")
PPO_vehicle_speed_train_energy_consumption_2.tolist()
PPO_vehicle_speed_train_energy_consumption_2=PPO_vehicle_speed_train_energy_consumption_2[:200]
PPO_vehicle_speed_train_ma_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/vehicle speed/PPO/30-40/20240517-143334/results/train_ma_energy_consumption.npy")
PPO_vehicle_speed_train_ma_energy_consumption_2.tolist()
PPO_vehicle_speed_train_ma_energy_consumption_2=PPO_vehicle_speed_train_ma_energy_consumption_2[:200]




PPO_vehicle_speed_train_rewards_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_rewards.npy")
PPO_vehicle_speed_train_rewards_3.tolist()
PPO_vehicle_speed_train_rewards_3=PPO_vehicle_speed_train_rewards_3[:200]
PPO_vehicle_speed_train_ma_rewards_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_rewards.npy")
PPO_vehicle_speed_train_ma_rewards_3.tolist()
PPO_vehicle_speed_train_ma_rewards_3=PPO_vehicle_speed_train_ma_rewards_3[:200]

PPO_vehicle_speed_train_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_completion_rate.npy")
PPO_vehicle_speed_train_completion_ratio_3.tolist()
PPO_vehicle_speed_train_completion_ratio_3=PPO_vehicle_speed_train_completion_ratio_3[:200]
PPO_vehicle_speed_train_ma_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_completion_rate.npy")
PPO_vehicle_speed_train_ma_completion_ratio_3.tolist()
PPO_vehicle_speed_train_ma_completion_ratio_3=PPO_vehicle_speed_train_ma_completion_ratio_3[:200]

PPO_vehicle_speed_train_delay_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_delay.npy")
PPO_vehicle_speed_train_delay_3.tolist()
PPO_vehicle_speed_train_delay_3=PPO_vehicle_speed_train_delay_3[:200]
PPO_vehicle_speed_train_ma_delay_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_delay.npy")
PPO_vehicle_speed_train_ma_delay_3.tolist()
PPO_vehicle_speed_train_ma_delay_3=PPO_vehicle_speed_train_ma_delay_3[:200]

PPO_vehicle_speed_train_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_energy_consumption.npy")
PPO_vehicle_speed_train_energy_consumption_3.tolist()
PPO_vehicle_speed_train_energy_consumption_3=PPO_vehicle_speed_train_energy_consumption_3[:200]
PPO_vehicle_speed_train_ma_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/convergence graph/0.0001/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_vehicle_speed_train_ma_energy_consumption_3.tolist()
PPO_vehicle_speed_train_ma_energy_consumption_3=PPO_vehicle_speed_train_ma_energy_consumption_3[:200]

###############################
# plot_vehicle_speed_completion_ratio_box_diagram(PPO_vehicle_speed_train_completion_ratio_1,PPO_vehicle_speed_train_completion_ratio_2,PPO_vehicle_speed_train_completion_ratio_3)
plot_vehicle_speed_delay_and_energy_consumption_bar_chart(PPO_vehicle_speed_train_delay_1,PPO_vehicle_speed_train_delay_2,PPO_vehicle_speed_train_delay_3,PPO_vehicle_speed_train_energy_consumption_1,PPO_vehicle_speed_train_energy_consumption_2,PPO_vehicle_speed_train_energy_consumption_3)
#
# plot_vehicle_speed_rewards(PPO_vehicle_speed_train_rewards_1,PPO_vehicle_speed_train_ma_rewards_1,PPO_vehicle_speed_train_rewards_2,PPO_vehicle_speed_train_ma_rewards_2,PPO_vehicle_speed_train_rewards_3,PPO_vehicle_speed_train_ma_rewards_3)
# plot_vehicle_speed_completion_ratio(PPO_vehicle_speed_train_ma_completion_ratio_1,PPO_vehicle_speed_train_ma_completion_ratio_2,PPO_vehicle_speed_train_ma_completion_ratio_3)
# plot_vehicle_speed_delay(PPO_vehicle_speed_train_ma_delay_1,PPO_vehicle_speed_train_ma_delay_2,PPO_vehicle_speed_train_ma_delay_3)
# plot_vehicle_speed_energy_consumption(PPO_vehicle_speed_train_ma_energy_consumption_1,PPO_vehicle_speed_train_ma_energy_consumption_2,PPO_vehicle_speed_train_ma_energy_consumption_3)
################################

#绘制不同最大跳数的收敛图
PPO_max_hop_number_train_rewards_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240806-124546/results/train_rewards.npy")
PPO_max_hop_number_train_rewards_1.tolist()
PPO_max_hop_number_train_rewards_1=PPO_max_hop_number_train_rewards_1[:200]
PPO_max_hop_number_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240806-124546/results/train_ma_rewards.npy")
PPO_max_hop_number_train_ma_rewards_1.tolist()
PPO_max_hop_number_train_ma_rewards_1=PPO_max_hop_number_train_ma_rewards_1[:200]

PPO_max_hop_number_train_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240806-124546/results/train_completion_rate.npy")
PPO_max_hop_number_train_completion_ratio_1.tolist()
PPO_max_hop_number_train_completion_ratio_1=PPO_max_hop_number_train_completion_ratio_1[:200]
PPO_max_hop_number_train_ma_completion_ratio_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240806-124546/results/train_ma_completion_rate.npy")
PPO_max_hop_number_train_ma_completion_ratio_1.tolist()
PPO_max_hop_number_train_ma_completion_ratio_1=PPO_max_hop_number_train_ma_completion_ratio_1[:200]

PPO_max_hop_number_train_delay_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240712-143702/results/train_delay.npy")
PPO_max_hop_number_train_delay_1.tolist()
PPO_max_hop_number_train_delay_1=PPO_max_hop_number_train_delay_1[:200]
PPO_max_hop_number_train_ma_delay_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240712-143702/results/train_ma_delay.npy")
PPO_max_hop_number_train_ma_delay_1.tolist()
PPO_max_hop_number_train_ma_delay_1=PPO_max_hop_number_train_ma_delay_1[:200]

PPO_max_hop_number_train_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240729-142941/results/train_energy_consumption.npy")
PPO_max_hop_number_train_energy_consumption_1.tolist()
PPO_max_hop_number_train_energy_consumption_1=PPO_max_hop_number_train_energy_consumption_1[:200]
PPO_max_hop_number_train_ma_energy_consumption_1 = np.load("C:/Users/23928/Desktop/result/hop number/1/5/20240729-142941/results/train_ma_energy_consumption.npy")
PPO_max_hop_number_train_ma_energy_consumption_1.tolist()
PPO_max_hop_number_train_ma_energy_consumption_1=PPO_max_hop_number_train_ma_energy_consumption_1[:200]




PPO_max_hop_number_train_rewards_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_rewards.npy")
PPO_max_hop_number_train_rewards_2.tolist()
PPO_max_hop_number_train_rewards_2=PPO_max_hop_number_train_rewards_2[:200]
PPO_max_hop_number_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_ma_rewards.npy")
PPO_max_hop_number_train_ma_rewards_2.tolist()
PPO_max_hop_number_train_ma_rewards_2=PPO_max_hop_number_train_ma_rewards_2[:200]

PPO_max_hop_number_train_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_completion_rate.npy")
PPO_max_hop_number_train_completion_ratio_2.tolist()
PPO_max_hop_number_train_completion_ratio_2=PPO_max_hop_number_train_completion_ratio_2[:200]
PPO_max_hop_number_train_ma_completion_ratio_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_ma_completion_rate.npy")
PPO_max_hop_number_train_ma_completion_ratio_2.tolist()
PPO_max_hop_number_train_ma_completion_ratio_2=PPO_max_hop_number_train_ma_completion_ratio_2[:200]

PPO_max_hop_number_train_delay_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_delay.npy")
PPO_max_hop_number_train_delay_2.tolist()
PPO_max_hop_number_train_delay_2=PPO_max_hop_number_train_delay_2[:200]
PPO_max_hop_number_train_ma_delay_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240710-211208/results/train_ma_delay.npy")
PPO_max_hop_number_train_ma_delay_2.tolist()
PPO_max_hop_number_train_ma_delay_2=PPO_max_hop_number_train_ma_delay_2[:200]

PPO_max_hop_number_train_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240816-184234-/results/train_energy_consumption.npy")
PPO_max_hop_number_train_energy_consumption_2.tolist()
PPO_max_hop_number_train_energy_consumption_2=PPO_max_hop_number_train_energy_consumption_2[:200]
PPO_max_hop_number_train_ma_energy_consumption_2 = np.load("C:/Users/23928/Desktop/result/hop number/2/5/20240816-184234-/results/train_ma_energy_consumption.npy")
PPO_max_hop_number_train_ma_energy_consumption_2.tolist()
PPO_max_hop_number_train_ma_energy_consumption_2=PPO_max_hop_number_train_ma_energy_consumption_2[:200]




PPO_max_hop_number_train_rewards_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240801-212705/results/train_rewards.npy")
PPO_max_hop_number_train_rewards_3.tolist()
PPO_max_hop_number_train_rewards_3=PPO_max_hop_number_train_rewards_3[:200]
PPO_max_hop_number_train_ma_rewards_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240801-212705/results/train_ma_rewards.npy")
PPO_max_hop_number_train_ma_rewards_3.tolist()
PPO_max_hop_number_train_ma_rewards_3=PPO_max_hop_number_train_ma_rewards_3[:200]

PPO_max_hop_number_train_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240801-212705/results/train_completion_rate.npy")
PPO_max_hop_number_train_completion_ratio_3.tolist()
PPO_max_hop_number_train_completion_ratio_3=PPO_max_hop_number_train_completion_ratio_3[:200]
PPO_max_hop_number_train_ma_completion_ratio_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240801-212705/results/train_ma_completion_rate.npy")
PPO_max_hop_number_train_ma_completion_ratio_3.tolist()
PPO_max_hop_number_train_ma_completion_ratio_3=PPO_max_hop_number_train_ma_completion_ratio_3[:200]

PPO_max_hop_number_train_delay_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240615-211912/results/train_delay.npy")
PPO_max_hop_number_train_delay_3.tolist()
PPO_max_hop_number_train_delay_3=PPO_max_hop_number_train_delay_3[:200]
PPO_max_hop_number_train_ma_delay_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240615-211912/results/train_ma_delay.npy")
PPO_max_hop_number_train_ma_delay_3.tolist()
PPO_max_hop_number_train_ma_delay_3=PPO_max_hop_number_train_ma_delay_3[:200]

PPO_max_hop_number_train_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240720-214205/results/train_energy_consumption.npy")
PPO_max_hop_number_train_energy_consumption_3.tolist()
PPO_max_hop_number_train_energy_consumption_3=PPO_max_hop_number_train_energy_consumption_3[:200]
PPO_max_hop_number_train_ma_energy_consumption_3 = np.load("C:/Users/23928/Desktop/result/hop number/3/5/20240720-214205/results/train_ma_energy_consumption.npy")
PPO_max_hop_number_train_ma_energy_consumption_3.tolist()
PPO_max_hop_number_train_ma_energy_consumption_3=PPO_max_hop_number_train_ma_energy_consumption_3[:200]




PPO_max_hop_number_train_rewards_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_rewards.npy")
PPO_max_hop_number_train_rewards_4.tolist()
PPO_max_hop_number_train_rewards_4=PPO_max_hop_number_train_rewards_4[:200]
PPO_max_hop_number_train_ma_rewards_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_ma_rewards.npy")
PPO_max_hop_number_train_ma_rewards_4.tolist()
PPO_max_hop_number_train_ma_rewards_4=PPO_max_hop_number_train_ma_rewards_4[:200]

PPO_max_hop_number_train_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_completion_rate.npy")
PPO_max_hop_number_train_completion_ratio_4.tolist()
PPO_max_hop_number_train_completion_ratio_4=PPO_max_hop_number_train_completion_ratio_4[:200]
PPO_max_hop_number_train_ma_completion_ratio_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_ma_completion_rate.npy")
PPO_max_hop_number_train_ma_completion_ratio_4.tolist()
PPO_max_hop_number_train_ma_completion_ratio_4=PPO_max_hop_number_train_ma_completion_ratio_4[:200]

PPO_max_hop_number_train_delay_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_delay.npy")
PPO_max_hop_number_train_delay_4.tolist()
PPO_max_hop_number_train_delay_4=PPO_max_hop_number_train_delay_4[:200]
PPO_max_hop_number_train_ma_delay_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_ma_delay.npy")
PPO_max_hop_number_train_ma_delay_4.tolist()
PPO_max_hop_number_train_ma_delay_4=PPO_max_hop_number_train_ma_delay_4[:200]

PPO_max_hop_number_train_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_energy_consumption.npy")
PPO_max_hop_number_train_energy_consumption_4.tolist()
PPO_max_hop_number_train_energy_consumption_4=PPO_max_hop_number_train_energy_consumption_4[:200]
PPO_max_hop_number_train_ma_energy_consumption_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/5/20240827-214752/results/train_ma_energy_consumption.npy")
PPO_max_hop_number_train_ma_energy_consumption_4.tolist()
PPO_max_hop_number_train_ma_energy_consumption_4=PPO_max_hop_number_train_ma_energy_consumption_4[:200]

# print(sum(PPO_max_hop_number_train_rewards_1)/len(PPO_max_hop_number_train_rewards_1),sum(PPO_max_hop_number_train_rewards_2)/len(PPO_max_hop_number_train_rewards_2),sum(PPO_max_hop_number_train_rewards_3)/len(PPO_max_hop_number_train_rewards_3),sum(PPO_max_hop_number_train_rewards_4)/len(PPO_max_hop_number_train_rewards_4))
# print(sum(PPO_max_hop_number_train_delay_1)/len(PPO_max_hop_number_train_delay_1),sum(PPO_max_hop_number_train_delay_2)/len(PPO_max_hop_number_train_delay_2),sum(PPO_max_hop_number_train_delay_3)/len(PPO_max_hop_number_train_delay_3),sum(PPO_max_hop_number_train_delay_4)/len(PPO_max_hop_number_train_delay_4))
# print(sum(PPO_max_hop_number_train_energy_consumption_1)/len(PPO_max_hop_number_train_energy_consumption_1),sum(PPO_max_hop_number_train_energy_consumption_2)/len(PPO_max_hop_number_train_energy_consumption_2),sum(PPO_max_hop_number_train_energy_consumption_3)/len(PPO_max_hop_number_train_energy_consumption_3),sum(PPO_max_hop_number_train_energy_consumption_4)/len(PPO_max_hop_number_train_energy_consumption_4))

# plot_hob_number_rewards(PPO_max_hop_number_train_rewards_1,PPO_max_hop_number_train_rewards_2,PPO_max_hop_number_train_rewards_3)
# plot_hob_number_delay(PPO_max_hop_number_train_delay_1,PPO_max_hop_number_train_delay_2,PPO_max_hop_number_train_delay_3)
# plot_hob_number_energy_consumption(PPO_max_hop_number_train_energy_consumption_1,PPO_max_hop_number_train_energy_consumption_2,PPO_max_hop_number_train_energy_consumption_3)









#绘制绘制beacon周期收敛图
PPO_train_rewards_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_rewards.npy")
PPO_train_rewards_beacon_cycle_1.tolist()
PPO_train_rewards_beacon_cycle_1=PPO_train_rewards_beacon_cycle_1[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_ma_rewards.npy")
PPO_train_ma_rewards_beacon_cycle_1.tolist()
PPO_train_ma_rewards_beacon_cycle_1=PPO_train_ma_rewards_beacon_cycle_1[:200]

PPO_train_completion_ratio_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_completion_rate.npy")
PPO_train_completion_ratio_beacon_cycle_1.tolist()
PPO_train_completion_ratio_beacon_cycle_1=PPO_train_completion_ratio_beacon_cycle_1[:200]
PPO_train_ma_completion_ratio_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_beacon_cycle_1.tolist()
PPO_train_ma_completion_ratio_beacon_cycle_1=PPO_train_ma_completion_ratio_beacon_cycle_1[:200]

PPO_train_delay_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_delay.npy")
PPO_train_delay_beacon_cycle_1.tolist()
PPO_train_delay_beacon_cycle_1=PPO_train_delay_beacon_cycle_1[:200]
PPO_train_ma_delay_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_ma_delay.npy")
PPO_train_ma_delay_beacon_cycle_1.tolist()
PPO_train_ma_delay_beacon_cycle_1=PPO_train_ma_delay_beacon_cycle_1[:200]

PPO_train_energy_consumption_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_energy_consumption.npy")
PPO_train_energy_consumption_beacon_cycle_1.tolist()
PPO_train_energy_consumption_beacon_cycle_1=PPO_train_energy_consumption_beacon_cycle_1[:200]
PPO_train_ma_energy_consumption_beacon_cycle_1 = np.load("C:/Users/23928/Desktop/result/beacon cycle/1/20240504-182006/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_beacon_cycle_1.tolist()
PPO_train_ma_energy_consumption_beacon_cycle_1=PPO_train_ma_energy_consumption_beacon_cycle_1[:200]



PPO_train_rewards_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_rewards.npy")
PPO_train_rewards_beacon_cycle_2.tolist()
PPO_train_rewards_beacon_cycle_2=PPO_train_rewards_beacon_cycle_2[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_ma_rewards.npy")
PPO_train_ma_rewards_beacon_cycle_2.tolist()
PPO_train_ma_rewards_beacon_cycle_2=PPO_train_ma_rewards_beacon_cycle_2[:200]

PPO_train_completion_ratio_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_completion_rate.npy")
PPO_train_completion_ratio_beacon_cycle_2.tolist()
PPO_train_completion_ratio_beacon_cycle_2=PPO_train_completion_ratio_beacon_cycle_2[:200]
PPO_train_ma_completion_ratio_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_beacon_cycle_2.tolist()
PPO_train_ma_completion_ratio_beacon_cycle_2=PPO_train_ma_completion_ratio_beacon_cycle_2[:200]

PPO_train_delay_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_delay.npy")
PPO_train_delay_beacon_cycle_2.tolist()
PPO_train_delay_beacon_cycle_2=PPO_train_delay_beacon_cycle_2[:200]
PPO_train_ma_delay_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_ma_delay.npy")
PPO_train_ma_delay_beacon_cycle_2.tolist()
PPO_train_ma_delay_beacon_cycle_2=PPO_train_ma_delay_beacon_cycle_2[:200]

PPO_train_energy_consumption_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_energy_consumption.npy")
PPO_train_energy_consumption_beacon_cycle_2.tolist()
PPO_train_energy_consumption_beacon_cycle_2=PPO_train_energy_consumption_beacon_cycle_2[:200]
PPO_train_ma_energy_consumption_beacon_cycle_2 = np.load("C:/Users/23928/Desktop/result/beacon cycle/2/20240717-163808/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_beacon_cycle_2.tolist()
PPO_train_ma_energy_consumption_beacon_cycle_2=PPO_train_ma_energy_consumption_beacon_cycle_2[:200]



PPO_train_rewards_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_rewards.npy")
PPO_train_rewards_beacon_cycle_3.tolist()
PPO_train_rewards_beacon_cycle_3=PPO_train_rewards_beacon_cycle_3[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_ma_rewards.npy")
PPO_train_ma_rewards_beacon_cycle_3.tolist()
PPO_train_ma_rewards_beacon_cycle_3=PPO_train_ma_rewards_beacon_cycle_3[:200]

PPO_train_completion_ratio_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_completion_rate.npy")
PPO_train_completion_ratio_beacon_cycle_3.tolist()
PPO_train_completion_ratio_beacon_cycle_3=PPO_train_completion_ratio_beacon_cycle_3[:200]
PPO_train_ma_completion_ratio_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_beacon_cycle_3.tolist()
PPO_train_ma_completion_ratio_beacon_cycle_3=PPO_train_ma_completion_ratio_beacon_cycle_3[:200]

PPO_train_delay_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_delay.npy")
PPO_train_delay_beacon_cycle_3.tolist()
PPO_train_delay_beacon_cycle_3=PPO_train_delay_beacon_cycle_3[:200]
PPO_train_ma_delay_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_ma_delay.npy")
PPO_train_ma_delay_beacon_cycle_3.tolist()
PPO_train_ma_delay_beacon_cycle_3=PPO_train_ma_delay_beacon_cycle_3[:200]

PPO_train_energy_consumption_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_energy_consumption.npy")
PPO_train_energy_consumption_beacon_cycle_3.tolist()
PPO_train_energy_consumption_beacon_cycle_3=PPO_train_energy_consumption_beacon_cycle_3[:200]
PPO_train_ma_energy_consumption_beacon_cycle_3 = np.load("C:/Users/23928/Desktop/result/beacon cycle/3/20240715-153459/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_beacon_cycle_3.tolist()
PPO_train_ma_energy_consumption_beacon_cycle_3=PPO_train_ma_energy_consumption_beacon_cycle_3[:200]



PPO_train_rewards_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_rewards.npy")#20240804-140448//20240806-174255/20240815-171911
PPO_train_rewards_beacon_cycle_4.tolist()
PPO_train_rewards_beacon_cycle_4=PPO_train_rewards_beacon_cycle_4[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_ma_rewards.npy")
PPO_train_ma_rewards_beacon_cycle_4.tolist()
PPO_train_ma_rewards_beacon_cycle_4=PPO_train_ma_rewards_beacon_cycle_4[:200]

PPO_train_completion_ratio_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_completion_rate.npy")
PPO_train_completion_ratio_beacon_cycle_4.tolist()
PPO_train_completion_ratio_beacon_cycle_4=PPO_train_completion_ratio_beacon_cycle_4[:200]
PPO_train_ma_completion_ratio_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_beacon_cycle_4.tolist()
PPO_train_ma_completion_ratio_beacon_cycle_4=PPO_train_ma_completion_ratio_beacon_cycle_4[:200]

PPO_train_delay_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_delay.npy")
PPO_train_delay_beacon_cycle_4.tolist()
PPO_train_delay_beacon_cycle_4=PPO_train_delay_beacon_cycle_4[:200]
PPO_train_ma_delay_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_ma_delay.npy")
PPO_train_ma_delay_beacon_cycle_4.tolist()
PPO_train_ma_delay_beacon_cycle_4=PPO_train_ma_delay_beacon_cycle_4[:200]

PPO_train_energy_consumption_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_energy_consumption.npy")
PPO_train_energy_consumption_beacon_cycle_4.tolist()
PPO_train_energy_consumption_beacon_cycle_4=PPO_train_energy_consumption_beacon_cycle_4[:200]
PPO_train_ma_energy_consumption_beacon_cycle_4 = np.load("C:/Users/23928/Desktop/result/beacon cycle/4/20240815-171911/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_beacon_cycle_4.tolist()
PPO_train_ma_energy_consumption_beacon_cycle_4=PPO_train_ma_energy_consumption_beacon_cycle_4[:200]



PPO_train_rewards_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_rewards.npy")
PPO_train_rewards_beacon_cycle_5.tolist()
PPO_train_rewards_beacon_cycle_5=PPO_train_rewards_beacon_cycle_5[:200]
# PPO_train_rewards_1=PPO_train_rewards_1[:200]
PPO_train_ma_rewards_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_ma_rewards.npy")
PPO_train_ma_rewards_beacon_cycle_5.tolist()
PPO_train_ma_rewards_beacon_cycle_5=PPO_train_ma_rewards_beacon_cycle_5[:200]

PPO_train_completion_ratio_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_completion_rate.npy")
PPO_train_completion_ratio_beacon_cycle_5.tolist()
PPO_train_completion_ratio_beacon_cycle_5=PPO_train_completion_ratio_beacon_cycle_5[:200]
PPO_train_ma_completion_ratio_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_ma_completion_rate.npy")
PPO_train_ma_completion_ratio_beacon_cycle_5.tolist()
PPO_train_ma_completion_ratio_beacon_cycle_5=PPO_train_ma_completion_ratio_beacon_cycle_5[:200]

PPO_train_delay_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_delay.npy")
PPO_train_delay_beacon_cycle_5.tolist()
PPO_train_delay_beacon_cycle_5=PPO_train_delay_beacon_cycle_5[:200]
PPO_train_ma_delay_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_ma_delay.npy")
PPO_train_ma_delay_beacon_cycle_5.tolist()
PPO_train_ma_delay_beacon_cycle_5=PPO_train_ma_delay_beacon_cycle_5[:200]

PPO_train_energy_consumption_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_energy_consumption.npy")
PPO_train_energy_consumption_beacon_cycle_5.tolist()
PPO_train_energy_consumption_beacon_cycle_5=PPO_train_energy_consumption_beacon_cycle_5[:200]
PPO_train_ma_energy_consumption_beacon_cycle_5 = np.load("C:/Users/23928/Desktop/result/beacon cycle/5/20240719-215703/results/train_ma_energy_consumption.npy")
PPO_train_ma_energy_consumption_beacon_cycle_5.tolist()
PPO_train_ma_energy_consumption_beacon_cycle_5=PPO_train_ma_energy_consumption_beacon_cycle_5[:200]


#
# plot_PPO_rewards_beacon_cycle(PPO_train_rewards_beacon_cycle_1,PPO_train_ma_rewards_beacon_cycle_1,PPO_train_rewards_beacon_cycle_2,PPO_train_ma_rewards_beacon_cycle_2,PPO_train_rewards_beacon_cycle_3,PPO_train_ma_rewards_beacon_cycle_3,PPO_train_rewards_beacon_cycle_4,PPO_train_ma_rewards_beacon_cycle_4,PPO_train_rewards_beacon_cycle_5,PPO_train_ma_rewards_beacon_cycle_5)
# plot_beacon_cycle_delay_bar_chart(PPO_train_delay_beacon_cycle_1,PPO_train_delay_beacon_cycle_2,PPO_train_delay_beacon_cycle_3,PPO_train_delay_beacon_cycle_4,PPO_train_delay_beacon_cycle_5)
# plot_beacon_cycle_energy_consumption_bar_chart(PPO_train_energy_consumption_beacon_cycle_1,PPO_train_energy_consumption_beacon_cycle_2,PPO_train_energy_consumption_beacon_cycle_3,PPO_train_energy_consumption_beacon_cycle_4,PPO_train_energy_consumption_beacon_cycle_5)


# plot_PPO_completion_ratio_beacon_cycle(PPO_train_completion_ratio_beacon_cycle_1,PPO_train_ma_completion_ratio_beacon_cycle_1,PPO_train_completion_ratio_beacon_cycle_2,PPO_train_ma_completion_ratio_beacon_cycle_2,PPO_train_completion_ratio_beacon_cycle_3,PPO_train_ma_completion_ratio_beacon_cycle_3,PPO_train_completion_ratio_beacon_cycle_4,PPO_train_ma_completion_ratio_beacon_cycle_4,PPO_train_completion_ratio_beacon_cycle_5,PPO_train_ma_completion_ratio_beacon_cycle_5)
# plot_PPO_delay_beacon_cycle(PPO_train_delay_beacon_cycle_1,PPO_train_ma_delay_beacon_cycle_1,PPO_train_delay_beacon_cycle_2,PPO_train_ma_delay_beacon_cycle_2,PPO_train_delay_beacon_cycle_3,PPO_train_ma_delay_beacon_cycle_3,PPO_train_delay_beacon_cycle_4,PPO_train_ma_delay_beacon_cycle_4,PPO_train_delay_beacon_cycle_5,PPO_train_ma_delay_beacon_cycle_5)
# plot_PPO_energy_consumption_beacon_cycle(PPO_train_energy_consumption_beacon_cycle_1,PPO_train_ma_energy_consumption_beacon_cycle_1,PPO_train_energy_consumption_beacon_cycle_2,PPO_train_ma_energy_consumption_beacon_cycle_2,PPO_train_energy_consumption_beacon_cycle_3,PPO_train_ma_energy_consumption_beacon_cycle_3,PPO_train_energy_consumption_beacon_cycle_4,PPO_train_ma_energy_consumption_beacon_cycle_4,PPO_train_energy_consumption_beacon_cycle_5,PPO_train_ma_energy_consumption_beacon_cycle_5)
# plot_beacon_cycle_delay_and_energy_consumption_bar_chart(PPO_train_delay_beacon_cycle_1,PPO_train_delay_beacon_cycle_2,PPO_train_delay_beacon_cycle_3,PPO_train_delay_beacon_cycle_4,PPO_train_delay_beacon_cycle_5,PPO_train_energy_consumption_beacon_cycle_1,PPO_train_energy_consumption_beacon_cycle_2,PPO_train_energy_consumption_beacon_cycle_3,PPO_train_energy_consumption_beacon_cycle_4,PPO_train_energy_consumption_beacon_cycle_5)

# #绘制绘制hop number收敛图
# PPO_train_rewards_hop_number_1 = np.load("")
# PPO_train_rewards_hop_number_1.tolist()
# PPO_train_rewards_hop_number_1=PPO_train_rewards_hop_number_1[:200]
# # PPO_train_rewards_1=PPO_train_rewards_1[:200]
# PPO_train_ma_rewards_hop_number_1 = np.load("")
# PPO_train_ma_rewards_hop_number_1.tolist()
# PPO_train_ma_rewards_hop_number_1=PPO_train_ma_rewards_hop_number_1[:200]
#
# PPO_train_completion_ratio_hop_number_1 = np.load("")
# PPO_train_completion_ratio_hop_number_1.tolist()
# PPO_train_completion_ratio_hop_number_1=PPO_train_completion_ratio_hop_number_1[:200]
# PPO_train_ma_completion_ratio_hop_number_1 = np.load("")
# PPO_train_ma_completion_ratio_hop_number_1.tolist()
# PPO_train_ma_completion_ratio_hop_number_1=PPO_train_ma_completion_ratio_hop_number_1[:200]
#
# PPO_train_delay_hop_number_1 = np.load("")
# PPO_train_delay_hop_number_1.tolist()
# PPO_train_delay_hop_number_1=PPO_train_delay_hop_number_1[:200]
# PPO_train_ma_delay_hop_number_1 = np.load("")
# PPO_train_ma_delay_hop_number_1.tolist()
# PPO_train_ma_delay_hop_number_1=PPO_train_ma_delay_hop_number_1[:200]
#
# PPO_train_energy_consumption_hop_number_1 = np.load("")
# PPO_train_energy_consumption_hop_number_1.tolist()
# PPO_train_energy_consumption_hop_number_1=PPO_train_energy_consumption_hop_number_1[:200]
# PPO_train_ma_energy_consumption_hop_number_1 = np.load("")
# PPO_train_ma_energy_consumption_hop_number_1.tolist()
# PPO_train_ma_energy_consumption_hop_number_1=PPO_train_ma_energy_consumption_hop_number_1[:200]
#
#
#
#
# PPO_train_rewards_hop_number_2 = np.load("")
# PPO_train_rewards_hop_number_2.tolist()
# PPO_train_rewards_hop_number_2=PPO_train_rewards_hop_number_2[:200]
# # PPO_train_rewards_1=PPO_train_rewards_1[:200]
# PPO_train_ma_rewards_hop_number_2 = np.load("")
# PPO_train_ma_rewards_hop_number_2.tolist()
# PPO_train_ma_rewards_hop_number_2=PPO_train_ma_rewards_hop_number_2[:200]
#
# PPO_train_completion_ratio_hop_number_2 = np.load("")
# PPO_train_completion_ratio_hop_number_2.tolist()
# PPO_train_completion_ratio_hop_number_2=PPO_train_completion_ratio_hop_number_2[:200]
# PPO_train_ma_completion_ratio_hop_number_2 = np.load("")
# PPO_train_ma_completion_ratio_hop_number_2.tolist()
# PPO_train_ma_completion_ratio_hop_number_2=PPO_train_ma_completion_ratio_hop_number_2[:200]
#
# PPO_train_delay_hop_number_2 = np.load("")
# PPO_train_delay_hop_number_2.tolist()
# PPO_train_delay_hop_number_2=PPO_train_delay_hop_number_2[:200]
# PPO_train_ma_delay_hop_number_2 = np.load("")
# PPO_train_ma_delay_hop_number_2.tolist()
# PPO_train_ma_delay_hop_number_2=PPO_train_ma_delay_hop_number_2[:200]
#
# PPO_train_energy_consumption_hop_number_2 = np.load("")
# PPO_train_energy_consumption_hop_number_2.tolist()
# PPO_train_energy_consumption_hop_number_2=PPO_train_energy_consumption_hop_number_2[:200]
# PPO_train_ma_energy_consumption_hop_number_2 = np.load("")
# PPO_train_ma_energy_consumption_hop_number_2.tolist()
# PPO_train_ma_energy_consumption_hop_number_2=PPO_train_ma_energy_consumption_hop_number_2[:200]
#
#
#
#
# PPO_train_rewards_hop_number_3 = np.load("")
# PPO_train_rewards_hop_number_3.tolist()
# PPO_train_rewards_hop_number_3=PPO_train_rewards_hop_number_3[:200]
# # PPO_train_rewards_1=PPO_train_rewards_1[:200]
# PPO_train_ma_rewards_hop_number_3 = np.load("")
# PPO_train_ma_rewards_hop_number_3.tolist()
# PPO_train_ma_rewards_hop_number_3=PPO_train_ma_rewards_hop_number_3[:200]
#
# PPO_train_completion_ratio_hop_number_3 = np.load("")
# PPO_train_completion_ratio_hop_number_3.tolist()
# PPO_train_completion_ratio_hop_number_3=PPO_train_completion_ratio_hop_number_3[:200]
# PPO_train_ma_completion_ratio_hop_number_3 = np.load("")
# PPO_train_ma_completion_ratio_hop_number_3.tolist()
# PPO_train_ma_completion_ratio_hop_number_3=PPO_train_ma_completion_ratio_hop_number_3[:200]
#
# PPO_train_delay_hop_number_3 = np.load("")
# PPO_train_delay_hop_number_3.tolist()
# PPO_train_delay_hop_number_3=PPO_train_delay_hop_number_3[:200]
# PPO_train_ma_delay_hop_number_3 = np.load("")
# PPO_train_ma_delay_hop_number_3.tolist()
# PPO_train_ma_delay_hop_number_3=PPO_train_ma_delay_hop_number_3[:200]
#
# PPO_train_energy_consumption_hop_number_3 = np.load("")
# PPO_train_energy_consumption_hop_number_3.tolist()
# PPO_train_energy_consumption_hop_number_3=PPO_train_energy_consumption_hop_number_3[:200]
# PPO_train_ma_energy_consumption_hop_number_3 = np.load("")
# PPO_train_ma_energy_consumption_hop_number_3.tolist()
# PPO_train_ma_energy_consumption_hop_number_3=PPO_train_ma_energy_consumption_hop_number_3[:200]
#
#
#
# PPO_train_rewards_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_rewards.npy")
# PPO_train_rewards_hop_number_4.tolist()
# PPO_train_rewards_hop_number_4=PPO_train_rewards_hop_number_4[:200]
# # PPO_train_rewards_1=PPO_train_rewards_1[:200]
# PPO_train_ma_rewards_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_ma_rewards.npy")
# PPO_train_ma_rewards_hop_number_4.tolist()
# PPO_train_ma_rewards_hop_number_4=PPO_train_ma_rewards_hop_number_4[:200]
#
# PPO_train_completion_ratio_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_completion_rate.npy")
# PPO_train_completion_ratio_hop_number_4.tolist()
# PPO_train_completion_ratio_hop_number_4=PPO_train_completion_ratio_hop_number_4[:200]
# PPO_train_ma_completion_ratio_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_ma_completion_rate.npy")
# PPO_train_ma_completion_ratio_hop_number_4.tolist()
# PPO_train_ma_completion_ratio_hop_number_4=PPO_train_ma_completion_ratio_hop_number_4[:200]
#
# PPO_train_delay_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_delay.npy")
# PPO_train_delay_hop_number_4.tolist()
# PPO_train_delay_hop_number_4=PPO_train_delay_hop_number_4[:200]
# PPO_train_ma_delay_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_ma_delay.npy")
# PPO_train_ma_delay_hop_number_4.tolist()
# PPO_train_ma_delay_hop_number_4=PPO_train_ma_delay_hop_number_4[:200]
#
# PPO_train_energy_consumption_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_energy_consumption.npy")
# PPO_train_energy_consumption_hop_number_4.tolist()
# PPO_train_energy_consumption_hop_number_4=PPO_train_energy_consumption_hop_number_4[:200]
# PPO_train_ma_energy_consumption_hop_number_4 = np.load("C:/Users/23928/Desktop/result/hop number/4/PPO/20240827-214752/results/train_ma_energy_consumption.npy")
# PPO_train_ma_energy_consumption_hop_number_4.tolist()
# PPO_train_ma_energy_consumption_hop_number_4=PPO_train_ma_energy_consumption_hop_number_4[:200]
#
# # plot_PPO_rewards_hop_number();