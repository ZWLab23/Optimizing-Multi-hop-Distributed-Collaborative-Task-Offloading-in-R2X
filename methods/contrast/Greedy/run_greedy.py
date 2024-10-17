
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
from env.environment_contrast_greedy import RoadState
from env.config_contrast import VehicularEnvConfig
from env.utils import plot_rewards,  save_results_1, plot_completion_rate, plot_delay, plot_energy_consumption
import dill as pickle  # 用dill代替pickle,保存参数的
from methods.contrast.Greedy.greedy_tasksize import Greedy


def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name',default='Greedy',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='Multihop-V2V-predict',type=str,help="name of environment")
    parser.add_argument('--train_eps', default=300, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=30, type=int, help="episodes of testing")
    parser.add_argument('--result_path',
                        default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/results/')
    parser.add_argument('--model_path',  # path to save models
                        default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    return args



def train(cfg, env, agent):
    """ Training """
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
        state,neighbor_vehicle_list,new_task_list = env.reset()
        old_task_list=new_task_list
        done = False
        ep_reward = 0
        ep_delay=0
        ep_energy_consumption=0
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number =0

        while not done:
            action = agent.choose_action(state,neighbor_vehicle_list,old_task_list)
            state_,neighbor_vehicle_list, reward,dealy ,energy_consumption,done,  offloading_rsu,offloading_vehicle, offloading_cloud, complete,new_task_list = env.step(action,old_task_list)
            steps += 1
            old_task_list = new_task_list
            ep_reward += reward
            ep_delay+=dealy
            ep_energy_consumption+=energy_consumption
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
        completion_rate=complete_number/((VehicularEnvConfig().rsu_number-2)*(VehicularEnvConfig().time_slot_end+1))
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





if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env = RoadState()
    agent = Greedy()
    train(cfg, env, agent)



