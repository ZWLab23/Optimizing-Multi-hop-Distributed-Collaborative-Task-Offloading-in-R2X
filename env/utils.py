#!/usr/bin/env python
# coding=utf-8
"""
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-07-13 22:15:46
Description:
Environment:
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties  # 导入字体模块
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
def chinese_font():
    """ 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体 """
    try:
        font = FontProperties(
            fname='C:/Windows/Fonts/STSONG.TTF', size=15)  # fname系统字体路径，此处是windows的
    except:
        font = None
    return font







def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=18, fontname='Times New Roman')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma_rewards')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()





def plot_delay(delay, ma_delay, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=18, fontname='Times New Roman')
    plt.plot(delay, label='delay')
    plt.plot(ma_delay, label='ma_delay')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_delay_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()


def plot_energy_consumption(energy_consumption, ma_energy_consumption, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('energy_consumption', fontsize=18, fontname='Times New Roman')
    plt.plot(energy_consumption, label='energy_consumption')
    plt.plot(ma_energy_consumption, label='ma_energy_consumption')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_energy_consumption_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()

def plot_backlogs(backlogs, ma_backlogs, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('backlogs', fontsize=18, fontname='Times New Roman')
    plt.plot(backlogs, label='backlogs')
    plt.plot(ma_backlogs, label='ma_backlogs')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_backlogs_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()



def save_results_1(dic, tag='train', path='./results'):
    """ 保存奖励 """
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    # print('Results saved！')





def plot_completion_rate(completion_rate, ma_completion_rate, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks( fontsize=16, fontname='Times New Roman')
    plt.yticks( fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=18, fontname='Times New Roman')
    plt.plot(completion_rate, label='completion_ratio')
    plt.plot(ma_completion_rate, label='ma_completion_ratio')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size':18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_completion_rate_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()




def plot_PPO_rewards(rewards):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(rewards)
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_rewards.pdf', format='pdf')
    plt.show()


def plot_PPO_completion_ratio(comptete):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('comptete ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(comptete)
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_completion_ratio.pdf', format='pdf')
    plt.show()









def plot_PPO_rewards_lr(PPO_train_rewards_1_lr,PPO_train_ma_rewards_1_lr,PPO_train_rewards_2_lr,PPO_train_ma_rewards_2_lr,PPO_train_rewards_3_lr,PPO_train_ma_rewards_3_lr):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks([0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks([-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')

    plt.plot(PPO_train_rewards_1_lr,color="#CDE4E4")
    plt.plot(PPO_train_rewards_2_lr,color="#B7D0EA")
    plt.plot(PPO_train_rewards_3_lr,color="#EBD2D6")

    plt.plot(PPO_train_ma_rewards_1_lr, label='lr=0.001',color="#84C2AE", linewidth=3)
    plt.plot(PPO_train_ma_rewards_2_lr, label='lr=0.0005',color="#407BD0", linewidth=3)
    plt.plot(PPO_train_ma_rewards_3_lr, label='lr=0.0001',color="#C24976", linewidth=3)

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    # # 填充置信区间
    # # 计算每个数据点的置信区间（假设为标准差）
    # confidence_interval_1 = np.std(PPO_train_ma_rewards_1_lr)
    # confidence_interval_2 = np.std(PPO_train_ma_rewards_2_lr)
    # confidence_interval_3 = np.std(PPO_train_ma_rewards_3_lr)

    # x = np.arange(len(PPO_train_ma_rewards_1_lr))
    # plt.fill_between(x, PPO_train_ma_rewards_1_lr - confidence_interval_1,
    #                  PPO_train_ma_rewards_1_lr + confidence_interval_1, alpha=0.2, color="#84C2AE")
    # plt.fill_between(x, PPO_train_ma_rewards_2_lr - confidence_interval_2,
    #                  PPO_train_ma_rewards_2_lr + confidence_interval_2, alpha=0.2, color="#407BD0")
    # plt.fill_between(x, PPO_train_ma_rewards_3_lr - confidence_interval_3,
    #                  PPO_train_ma_rewards_3_lr + confidence_interval_3, alpha=0.2, color="#C24976")

    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_rewards_lr.pdf', format='pdf')
    plt.show()


def plot_PPO_completion_ratio_lr(PPO_train_completion_ratio_1_lr,PPO_train_ma_completion_ratio_1_lr,PPO_train_completion_ratio_2_lr,PPO_train_ma_completion_ratio_2_lr,PPO_train_completion_ratio_3_lr,PPO_train_ma_completion_ratio_3_lr):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(PPO_train_completion_ratio_1_lr,color="#CDE4E4")
    plt.plot(PPO_train_completion_ratio_2_lr,color="#B7D0EA")
    plt.plot(PPO_train_completion_ratio_3_lr,color="#EBD2D6")
    plt.plot(PPO_train_ma_completion_ratio_1_lr, label='lr=0.001',color="#84C2AE", linewidth=3)
    plt.plot(PPO_train_ma_completion_ratio_2_lr, label='lr=0.0005',color="#407BD0", linewidth=3)
    plt.plot(PPO_train_ma_completion_ratio_3_lr, label='lr=0.0001',color="#C24976", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_completion_ratio_lr.pdf', format='pdf')
    plt.show()




def plot_PPO_delay_lr(PPO_train_delay_1_lr,PPO_train_ma_delay_1_lr,PPO_train_delay_2_lr,PPO_train_ma_delay_2_lr,PPO_train_delay_3_lr,PPO_train_ma_delay_3_lr):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay (s)', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( [0,400,800,1200,1600,2000],fontsize=22, fontname='Times New Roman')
    plt.plot(PPO_train_delay_1_lr,color="#CDE4E4")
    plt.plot(PPO_train_delay_2_lr,color="#B7D0EA")
    plt.plot(PPO_train_delay_3_lr,color="#EBD2D6")
    plt.plot(PPO_train_ma_delay_1_lr, label='lr=0.001',color="#84C2AE", linewidth=3)
    plt.plot(PPO_train_ma_delay_2_lr, label='lr=0.0005',color="#407BD0", linewidth=3)
    plt.plot(PPO_train_ma_delay_3_lr, label='lr=0.0001',color="#C24976", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_delay_lr.pdf', format='pdf')
    plt.show()



def plot_PPO_energy_consumption_lr(PPO_train_energy_consumption_1_lr,PPO_train_ma_energy_consumption_1_lr,PPO_train_energy_consumption_2_lr,PPO_train_ma_energy_consumption_2_lr,PPO_train_energy_consumption_3_lr,PPO_train_ma_energy_consumption_3_lr):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption (J)', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(PPO_train_energy_consumption_1_lr,color="#CDE4E4")
    plt.plot(PPO_train_energy_consumption_2_lr,color="#B7D0EA")
    plt.plot(PPO_train_energy_consumption_3_lr,color="#EBD2D6")
    plt.plot(PPO_train_ma_energy_consumption_1_lr, label='lr=0.001',color="#84C2AE", linewidth=3)
    plt.plot(PPO_train_ma_energy_consumption_2_lr, label='lr=0.0005',color="#407BD0", linewidth=3)
    plt.plot(PPO_train_ma_energy_consumption_3_lr, label='lr=0.0001',color="#C24976", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_energy_consumption_lr.pdf', format='pdf')
    plt.show()






def plot_contrast_delay(PPO_delay,PPO_ma_delay,SAC_delay,SAC_ma_delay,Greedy_delay,Greedy_ma_delay):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay (s)', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(PPO_delay,color="#EBD2D6")
    plt.plot(SAC_delay,color="#B7D0EA")
    plt.plot(Greedy_delay,color="#CDE4E4")
    plt.plot(PPO_ma_delay, label='Our approach',color="#C24976", linewidth=3)
    plt.plot(SAC_ma_delay, label='LS',color="#407BD0", linewidth=3)
    plt.plot(Greedy_ma_delay, label='Greedy',color="#84C2AE", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('contrast_delay.pdf', format='pdf')
    plt.show()

def plot_contrast_completion_ratio(PPO_complete,PPO_ma_complete,SAC_complete,SAC_ma_complete,Greedy_complete,Greedy_ma_complete):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(PPO_complete,color="#EBD2D6")
    plt.plot(SAC_complete,color="#B7D0EA")
    plt.plot(Greedy_complete,color="#CDE4E4")
    plt.plot(PPO_ma_complete, label='Our approach', color="#C24976", linewidth=3)
    plt.plot(SAC_ma_complete, label='LS', color="#407BD0", linewidth=3)
    plt.plot(Greedy_ma_complete, label='Greedy', color="#84C2AE", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('contrast_completion_ratio.pdf', format='pdf')
    plt.show()







def plot_tasksize_rewards(PPO_tasksize_2_rewards,PPO_tasksize_2_ma_rewards,PPO_tasksize_3_rewards,PPO_tasksize_3_ma_rewards,PPO_tasksize_4_rewards,PPO_tasksize_4_ma_rewards):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_tasksize_2_ma_rewards, label='tasksize=2')
    plt.plot(PPO_tasksize_3_ma_rewards, label='tasksize=3')
    plt.plot(PPO_tasksize_4_ma_rewards, label='tasksize=4')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_tasksize_rewards.pdf', format='pdf')
    plt.show()


def plot_tasksize_completion_ratio(PPO_tasksize_2_completion_ratio,PPO_tasksize_2_ma_completion_ratio,PPO_tasksize_3_completion_ratio,PPO_tasksize_3_ma_completion_ratio,PPO_tasksize_4_completion_ratio,PPO_tasksize_4_ma_completion_ratio):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_3_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_4_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_5_completion_ratio,alpha=0.2)
    plt.plot(PPO_tasksize_2_ma_completion_ratio, label='tasksize=2')
    plt.plot(PPO_tasksize_3_ma_completion_ratio, label='tasksize=3')
    plt.plot(PPO_tasksize_4_ma_completion_ratio, label='tasksize=4')
    # plt.plot(PPO_tasksize_5_ma_completion_ratio, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO__tasksize_completion_ratio.pdf', format='pdf')
    plt.show()

def plot_tasksize_delay(PPO_tasksize_2_delay,PPO_tasksize_2_ma_delay,PPO_tasksize_3_delay,PPO_tasksize_3_ma_delay,PPO_tasksize_4_delay,PPO_tasksize_4_ma_delay):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_tasksize_2_ma_delay, label='tasksize=2')
    plt.plot(PPO_tasksize_3_ma_delay, label='tasksize=3')
    plt.plot(PPO_tasksize_4_ma_delay, label='tasksize=4')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_tasksize_delay.pdf', format='pdf')
    plt.show()

def plot_tasksize_energy_consumption(PPO_tasksize_2_energy_consumption,PPO_tasksize_2_ma_energy_consumption,PPO_tasksize_3_energy_consumption,PPO_tasksize_3_ma_energy_consumption,PPO_tasksize_4_energy_consumption,PPO_tasksize_4_ma_energy_consumption):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_3_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_4_completion_ratio,alpha=0.2)
    # plt.plot(PPO_tasksize_5_completion_ratio,alpha=0.2)
    plt.plot(PPO_tasksize_2_ma_energy_consumption, label='tasksize=2')
    plt.plot(PPO_tasksize_3_ma_energy_consumption, label='tasksize=3')
    plt.plot(PPO_tasksize_4_ma_energy_consumption, label='tasksize=4')
    # plt.plot(PPO_tasksize_5_ma_completion_ratio, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_tasksize_energy_consumption.pdf', format='pdf')
    plt.show()



def plot_tasksize_delay_line_chart(delay_1,delay_2,delay_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('task size', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    x_datasize = [2, 2.5, 3]
    y_datasize=[delay_1,delay_2,delay_3]
    plt.plot(x_datasize, y_datasize ,linestyle='-', marker='o', markersize=8)
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('plot_tasksize_delay_line_chart.pdf', format='pdf')
    plt.show()

def plot_tasksize_energy_consumption_line_chart(energy_consumption_1,energy_consumption_2,energy_consumption_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('task size', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    x_datasize = [2, 2.5, 3]
    y_datasize=[energy_consumption_1,energy_consumption_2,energy_consumption_3]
    plt.plot(x_datasize, y_datasize ,linestyle='-', marker='o', markersize=8)
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('plot_tasksize_energy_consumption_line_chart.pdf', format='pdf')
    plt.show()


def plot_tasksize_rewards_box_diagram(PPO_tasksize_2_rewards, PPO_tasksize_3_rewards, PPO_tasksize_4_rewards):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task size',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('rewards',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_tasksize_2_rewards, PPO_tasksize_3_rewards, PPO_tasksize_4_rewards], patch_artist=True, labels=['2', '3', '4'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_tasksize_rewards_box_diagram.pdf', format='pdf')
    plt.show()

def plot_tasksize_completion_ratio_box_diagram(PPO_tasksize_2_completion_ratio, PPO_tasksize_3_completion_ratio, PPO_tasksize_4_completion_ratio):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task size',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('completion ratio',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_tasksize_2_completion_ratio, PPO_tasksize_3_completion_ratio, PPO_tasksize_4_completion_ratio], patch_artist=True, labels=['2', '3', '4'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_tasksize_completion_ratio_box_diagram.pdf', format='pdf')
    plt.show()


def plot_tasksize_delay_box_diagram(PPO_tasksize_2_delay, PPO_tasksize_3_delay, PPO_tasksize_4_delay):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task size',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('delay (s)',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_tasksize_2_delay, PPO_tasksize_3_delay, PPO_tasksize_4_delay], patch_artist=True, labels=['2', '3', '4'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_tasksize_delay_box_diagram.pdf', format='pdf')
    plt.show()

def plot_tasksize_energy_consumption_box_diagram(PPO_tasksize_2_energy_consumption, PPO_tasksize_3_energy_consumption, PPO_tasksize_4_energy_consumption):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task size',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('energy consumption (J)',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_tasksize_2_energy_consumption, PPO_tasksize_3_energy_consumption, PPO_tasksize_4_energy_consumption], patch_artist=True, labels=['2', '3', '4'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_tasksize_energy_consumption_box_diagram.pdf', format='pdf')
    plt.show()

def plot_computing_resource_rewards(PPO_computing_resource_rewards_1,PPO_computing_resource_rewards_2,PPO_computing_resource_rewards_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_computing_resource_rewards_1, label='task computation intensity=300')
    plt.plot(PPO_computing_resource_rewards_2, label='task computation intensity=325')
    plt.plot(PPO_computing_resource_rewards_3, label='task computation intensity=350')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_computing_resource_rewards.pdf', format='pdf')
    plt.show()



def plot_computing_resource_completion_ratio(PPO_computing_resource_completion_ratio_1,PPO_computing_resource_completion_ratio_2,PPO_computing_resource_completion_ratio_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_computing_resource_completion_ratio_1, label='task computation intensity=300')
    plt.plot(PPO_computing_resource_completion_ratio_2, label='task computation intensity=325')
    plt.plot(PPO_computing_resource_completion_ratio_3, label='task computation intensity=350')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_computing_resource_completion_ratio.pdf', format='pdf')
    plt.show()


def plot_computing_resource_delay(PPO_computing_resource_delay_1,PPO_computing_resource_delay_2,PPO_computing_resource_delay_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_computing_resource_delay_1, label='task computation intensity=300')
    plt.plot(PPO_computing_resource_delay_2, label='task computation intensity=325')
    plt.plot(PPO_computing_resource_delay_3, label='task computation intensity=350')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_computing_resource_delay.pdf', format='pdf')
    plt.show()

def plot_computing_resource_energy_consumption(PPO_computing_resource_energy_consumption_1,PPO_computing_resource_energy_consumption_2,PPO_computing_resource_energy_consumption_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(PPO_computing_resource_energy_consumption_1, label='task computation intensity=300')
    plt.plot(PPO_computing_resource_energy_consumption_2, label='task computation intensity=325')
    plt.plot(PPO_computing_resource_energy_consumption_3, label='task computation intensity=350')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_computing_resource_energy_consumption.pdf', format='pdf')
    plt.show()


def plot_computing_resource_rewards_box_diagram(PPO_computing_resource_rewards_1,PPO_computing_resource_rewards_2,PPO_computing_resource_rewards_3):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task computation intensity',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('rewards',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( [-20000,-15000,-10000,-5000,0],fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_computing_resource_rewards_1,PPO_computing_resource_rewards_2,PPO_computing_resource_rewards_3], patch_artist=True, labels=['300', '325', '350'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_computing_resource_rewards_box_diagram.pdf', format='pdf')
    plt.show()

def plot_computing_resource_completion_ratio_box_diagram(PPO_computing_resource_completion_ratio_1,PPO_computing_resource_completion_ratio_2,PPO_computing_resource_completion_ratio_3):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task computation intensity',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('completion ratio',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_computing_resource_completion_ratio_1,PPO_computing_resource_completion_ratio_2,PPO_computing_resource_completion_ratio_3], patch_artist=True, labels=['300', '325', '350'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_computing_resource_completion_ratio_box_diagram.pdf', format='pdf')
    plt.show()



def plot_computing_resource_delay_box_diagram(PPO_computing_resource_delay_1,PPO_computing_resource_delay_2,PPO_computing_resource_delay_3):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task computation intensity',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('delay',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( [0,500,1000,1500,2000],fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_computing_resource_delay_1,PPO_computing_resource_delay_2,PPO_computing_resource_delay_3], patch_artist=True, labels=['300', '325', '350'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_computing_resource_delay_box_diagram.pdf', format='pdf')
    plt.show()



def plot_computing_resource_energy_consumption_box_diagram(PPO_computing_resource_energy_consumption_1,PPO_computing_resource_energy_consumption_2,PPO_computing_resource_energy_consumption_3):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('task computation intensity',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('energy consumption',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([PPO_computing_resource_energy_consumption_1,PPO_computing_resource_energy_consumption_2,PPO_computing_resource_energy_consumption_3], patch_artist=True, labels=['300', '325', '350'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_computing_resource_energy_consumption_box_diagram.pdf', format='pdf')
    plt.show()









def plot_vehicle_number_rewards(vehicle_number_rewards_6,vehicle_number_ma_rewards_6,vehicle_number_rewards_8,vehicle_number_ma_rewards_8,vehicle_number_rewards_10,vehicle_number_ma_rewards_10):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( [-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')
    plt.plot(vehicle_number_rewards_6,color="#CDE4E4")
    plt.plot(vehicle_number_rewards_8,color="#B7D0EA")
    plt.plot(vehicle_number_rewards_10,color="#EBD2D6")
    plt.plot(vehicle_number_ma_rewards_6, label='vehicle number=6', color="#84C2AE", linewidth=3)
    plt.plot(vehicle_number_ma_rewards_8, label='vehicle number=8', color="#407BD0", linewidth=3)
    plt.plot(vehicle_number_ma_rewards_10, label='vehicle number=10', color="#C24976", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_number_rewards.pdf', format='pdf')
    plt.show()

def plot_vehicle_number_completion_ratio(vehicle_number_completion_ratio_6,vehicle_number_completion_ratio_8,vehicle_number_completion_ratio_10):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_number_completion_ratio_6, label='vehicle number=6', color="#84C2AE")
    plt.plot(vehicle_number_completion_ratio_8, label='vehicle number=8', color="#407BD0")
    plt.plot(vehicle_number_completion_ratio_10, label='vehicle number=10', color="#C24976")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_number_completion_ratio.pdf', format='pdf')
    plt.show()


def plot_vehicle_number_delay(vehicle_number_delay_6,vehicle_number_delay_8,vehicle_number_delay_10):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_number_delay_6, label='vehicle number=6')
    plt.plot(vehicle_number_delay_8, label='vehicle number=8')
    plt.plot(vehicle_number_delay_10, label='vehicle number=10')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_number_delay.pdf', format='pdf')
    plt.show()


def plot_vehicle_number_energy_consumption(vehicle_number_energy_consumption_6,vehicle_number_energy_consumption_8,vehicle_number_energy_consumption_10):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_number_energy_consumption_6, label='vehicle number=6')
    plt.plot(vehicle_number_energy_consumption_8, label='vehicle number=8')
    plt.plot(vehicle_number_energy_consumption_10, label='vehicle number=10')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_number_energy_consumption.pdf', format='pdf')
    plt.show()


def plot_vehicle_number_delay_and_energy_consumption_bar_chart(vehicle_number_delay_6,vehicle_number_delay_8,vehicle_number_delay_10,vehicle_number_energy_consumption_6,vehicle_number_energy_consumption_8,vehicle_number_energy_consumption_10):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('vehicle number', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average value', fontsize=26, fontname='Times New Roman')
    vehicle_number_average_delay_6=sum(vehicle_number_delay_6)/len(vehicle_number_delay_6)
    vehicle_number_average_delay_8 = sum(vehicle_number_delay_8) / len(vehicle_number_delay_8)
    vehicle_number_average_delay_10 = sum(vehicle_number_delay_10) / len(vehicle_number_delay_10)
    vehicle_number_average_energy_consumption_6=sum(vehicle_number_energy_consumption_6)/len(vehicle_number_energy_consumption_6)
    vehicle_number_average_energy_consumption_8 = sum(vehicle_number_energy_consumption_8) / len(vehicle_number_energy_consumption_8)
    vehicle_number_average_energy_consumption_10 = sum(vehicle_number_energy_consumption_10) / len(vehicle_number_energy_consumption_10)
    delay=[vehicle_number_average_delay_6,vehicle_number_average_delay_8,vehicle_number_average_delay_10]
    energy_consumption=[vehicle_number_average_energy_consumption_6,vehicle_number_average_energy_consumption_8,vehicle_number_average_energy_consumption_10]

    Width = 0.2
    x1 = np.arange(len(delay))
    x2=[x + Width for x in x1]
    plt.bar(x1, delay,label='average delay (s)',width=Width,color='#C3CEE4')
    plt.bar(x2, energy_consumption, label='average energy consumption (J)',width=Width,color='#E8D6B6')

    plt.xticks([r + Width/2 for r in range(len(delay))], ['6','8','10'])

    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( [200,400,600,800,1000,1200,1400,1600,1800,2000],fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_number_delay_and_energy_consumption_bar_chart.pdf', format='pdf')
    plt.show()



def plot_vehicle_number_completion_ratio_box_diagram(vehicle_number_completion_ratio_6,vehicle_number_completion_ratio_8,vehicle_number_completion_ratio_10):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('vehicle number',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('completion ratio',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([vehicle_number_completion_ratio_6,vehicle_number_completion_ratio_8,vehicle_number_completion_ratio_10], patch_artist=True, labels=['6', '8', '10'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_vehicle_number_completion_ratio_box_diagram.pdf', format='pdf')
    plt.show()


def plot_vehicle_number_delay_box_diagram(vehicle_number_delay_6,vehicle_number_delay_8,vehicle_number_delay_10):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('vehicle number',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('delay',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([vehicle_number_delay_6,vehicle_number_delay_8,vehicle_number_delay_10], patch_artist=True, labels=['6', '8', '10'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_vehicle_number_delay_box_diagram.pdf', format='pdf')
    plt.show()

def plot_vehicle_number_energy_consumption_box_diagram(vehicle_number_energy_consumption_6,vehicle_number_energy_consumption_8,vehicle_number_energy_consumption_10):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('vehicle number',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('energy consumption',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([vehicle_number_energy_consumption_6,vehicle_number_energy_consumption_8,vehicle_number_energy_consumption_10], patch_artist=True, labels=['6', '8', '10'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_vehicle_number_energy_consumption_box_diagram.pdf', format='pdf')
    plt.show()



def plot_vehicle_range_rewards(vehicle_range_rewards_100,vehicle_range_ma_rewards_100,vehicle_range_rewards_125,vehicle_range_ma_rewards_125,vehicle_range_rewards_150,vehicle_range_ma_rewards_150):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( [-20000,-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')
    plt.plot(vehicle_range_rewards_100,color="#CDE4E4")
    plt.plot(vehicle_range_rewards_125,color="#B7D0EA")
    plt.plot(vehicle_range_rewards_150,color="#EBD2D6")

    plt.plot(vehicle_range_ma_rewards_100, label='vehicle communication range=100', color="#84C2AE",linewidth=3)
    plt.plot(vehicle_range_ma_rewards_125, label='vehicle communication range=125', color="#407BD0",linewidth=3)
    plt.plot(vehicle_range_ma_rewards_150, label='vehicle communication range=150', color="#C24976",linewidth=3)

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_rewards.pdf', format='pdf')
    plt.show()

def plot_vehicle_range_completion_ratio(vehicle_range_completion_ratio_100,vehicle_range_completion_ratio_125,vehicle_range_completion_ratio_150):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(vehicle_range_completion_ratio_100, label='vehicle communication range=100', color="#84C2AE")
    plt.plot(vehicle_range_completion_ratio_125, label='vehicle communication range=125', color="#407BD0")
    plt.plot(vehicle_range_completion_ratio_150, label='vehicle communication range=150', color="#C24976")

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_completion_ratio.pdf', format='pdf')
    plt.show()


def plot_vehicle_range_delay(vehicle_range_delay_100,vehicle_range_delay_125,vehicle_range_delay_150):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_range_delay_100, label='vehicle communication range=100')
    plt.plot(vehicle_range_delay_125, label='vehicle communication range=125')
    plt.plot(vehicle_range_delay_150, label='vehicle communication range=150')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_delay.pdf', format='pdf')
    plt.show()


def plot_vehicle_range_energy_consumption(vehicle_range_energy_consumption_100,vehicle_range_energy_consumption_125,vehicle_range_energy_consumption_150):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_range_energy_consumption_100, label='vehicle communication range=100')
    plt.plot(vehicle_range_energy_consumption_125, label='vehicle communication range=125')
    plt.plot(vehicle_range_energy_consumption_150, label='vehicle communication range=150')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_energy_consumption.pdf', format='pdf')
    plt.show()


def plot_vehicle_range_completion_ratio_box_diagram(vehicle_range_100,vehicle_range_125,vehicle_range_150):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('vehicle communication range',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('completion ratio',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([vehicle_range_100,vehicle_range_125,vehicle_range_150], patch_artist=True, labels=['100', '125', '150'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_vehicle_range_completion_ratio_box_diagram.pdf', format='pdf')
    plt.show()


def plot_vehicle_range_delay_and_energy_consumption_bar_chart(vehicle_range_delay_100,vehicle_range_delay_125,vehicle_range_delay_150,vehicle_range_energy_consumption_100,vehicle_range_energy_consumption_125,vehicle_range_energy_consumption_150):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('vehicle communication range', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average value', fontsize=26, fontname='Times New Roman')
    vehicle_range_average_delay_100=sum(vehicle_range_delay_100)/len(vehicle_range_delay_100)
    vehicle_range_average_delay_125 = sum(vehicle_range_delay_125) / len(vehicle_range_delay_125)
    vehicle_range_average_delay_150 = sum(vehicle_range_delay_150) / len(vehicle_range_delay_150)
    vehicle_range_average_energy_consumption_100=sum(vehicle_range_energy_consumption_100)/len(vehicle_range_energy_consumption_100)
    vehicle_range_average_energy_consumption_125 = sum(vehicle_range_energy_consumption_125) / len(vehicle_range_energy_consumption_125)
    vehicle_range_average_energy_consumption_150 = sum(vehicle_range_energy_consumption_150) / len(vehicle_range_energy_consumption_150)
    delay=[vehicle_range_average_delay_100,vehicle_range_average_delay_125,vehicle_range_average_delay_150]
    energy_consumption=[vehicle_range_average_energy_consumption_100,vehicle_range_average_energy_consumption_125,vehicle_range_average_energy_consumption_150]

    Width = 0.2
    x1 = np.arange(len(delay))
    x2=[x + Width for x in x1]
    plt.bar(x1, delay,label='average delay (s)',width=Width,color='#C3CEE4')
    plt.bar(x2, energy_consumption, label='average energy consumption (J)',width=Width,color='#E8D6B6')

    plt.xticks([r + Width/2 for r in range(len(delay))], ['100','125','150'])

    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_delay_and_energy_consumption_bar_chart.pdf', format='pdf')
    plt.show()






def plot_vehicle_speed_rewards(vehicle_speed_rewards_1,vehicle_speed_ma_rewards_1,vehicle_speed_rewards_2,vehicle_speed_ma_rewards_2,vehicle_speed_rewards_3,vehicle_speed_ma_rewards_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( [-20000,-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')

    plt.plot(vehicle_speed_rewards_1,color="#CDE4E4")
    plt.plot(vehicle_speed_rewards_2,color="#B7D0EA")
    plt.plot(vehicle_speed_rewards_3,color="#EBD2D6")
    plt.plot(vehicle_speed_ma_rewards_1, label='vehicle speed=20-30', color="#84C2AE",linewidth=3)
    plt.plot(vehicle_speed_ma_rewards_2, label='vehicle speed=30-40', color="#407BD0",linewidth=3)
    plt.plot(vehicle_speed_ma_rewards_3, label='vehicle speed=40-50', color="#C24976",linewidth=3)

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_speed_rewards.pdf', format='pdf')
    plt.show()

def plot_vehicle_speed_completion_ratio(vehicle_speed_completion_ratio_1,vehicle_speed_completion_ratio_2,vehicle_speed_completion_ratio_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_speed_completion_ratio_1, label='vehicle speed=20-30')
    plt.plot(vehicle_speed_completion_ratio_2, label='vehicle speed=30-40')
    plt.plot(vehicle_speed_completion_ratio_3, label='vehicle speed=40-50')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_speed_completion_ratio.pdf', format='pdf')
    plt.show()


def plot_vehicle_speed_delay(vehicle_speed_delay_1,vehicle_speed_delay_2,vehicle_speed_delay_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_speed_delay_1, label='vehicle speed=20-30')
    plt.plot(vehicle_speed_delay_2, label='vehicle speed=30-40')
    plt.plot(vehicle_speed_delay_3, label='vehicle speed=40-50')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_speed_delay.pdf', format='pdf')
    plt.show()


def plot_vehicle_speed_energy_consumption(vehicle_speed_energy_consumption_1,vehicle_speed_energy_consumption_2,vehicle_speed_energy_consumption_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(vehicle_speed_energy_consumption_1, label='vehicle speed=20-30')
    plt.plot(vehicle_speed_energy_consumption_2, label='vehicle speed=30-40')
    plt.plot(vehicle_speed_energy_consumption_3, label='vehicle speed=40-50')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_range_speed_consumption.pdf', format='pdf')
    plt.show()




def plot_vehicle_speed_completion_ratio_box_diagram(vehicle_speed_completion_ratio_1,vehicle_speed_completion_ratio_2,vehicle_speed_completion_ratio_3):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.xlabel('vehicle speed',fontsize=26, fontname='Times New Roman')  # 添加横坐标标签
    plt.ylabel('completion ratio',fontsize=26, fontname='Times New Roman')  # 添加纵坐标标签
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    # 绘制箱型图

    bp = plt.boxplot([vehicle_speed_completion_ratio_1,vehicle_speed_completion_ratio_2,vehicle_speed_completion_ratio_3], patch_artist=True, labels=['20-30', '30-40', '40-50'])

    # 指定每个箱线图的颜色
    colors = ['#899CCB', '#F1C0C4', '#FBE7C0']  # 自定义颜色列表
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)  # 设置箱体颜色
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig('plot_vehicle_speed_completion_ratio_box_diagram.pdf', format='pdf')
    plt.show()


def plot_vehicle_speed_delay_and_energy_consumption_bar_chart(vehicle_speed_delay_1,vehicle_speed_delay_2,vehicle_speed_delay_3,vehicle_speed_energy_consumption_1,vehicle_speed_energy_consumption_2,vehicle_speed_energy_consumption_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('vehicle speed', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average value', fontsize=26, fontname='Times New Roman')
    vehicle_speed_average_delay_1=sum(vehicle_speed_delay_1)/len(vehicle_speed_delay_1)
    vehicle_speed_average_delay_2 = sum(vehicle_speed_delay_2) / len(vehicle_speed_delay_2)
    vehicle_speed_average_delay_3 = sum(vehicle_speed_delay_3) / len(vehicle_speed_delay_3)
    vehicle_speed_average_energy_consumption_1=sum(vehicle_speed_energy_consumption_1)/len(vehicle_speed_energy_consumption_1)
    vehicle_speed_average_energy_consumption_2 = sum(vehicle_speed_energy_consumption_2) / len(vehicle_speed_energy_consumption_2)
    vehicle_speed_average_energy_consumption_3 = sum(vehicle_speed_energy_consumption_3) / len(vehicle_speed_energy_consumption_3)
    delay=[vehicle_speed_average_delay_1,vehicle_speed_average_delay_2,vehicle_speed_average_delay_3]
    energy_consumption=[vehicle_speed_average_energy_consumption_1,vehicle_speed_average_energy_consumption_2,vehicle_speed_average_energy_consumption_3]

    Width = 0.2
    x1 = np.arange(len(delay))
    x2=[x + Width for x in x1]
    plt.bar(x1, delay,label='average delay (s)',width=Width,color='#C3CEE4')
    plt.bar(x2, energy_consumption, label='average energy consumption (J)',width=Width,color='#E8D6B6')

    plt.xticks([r + Width/2 for r in range(len(delay))], ['20-30','30-40','40-50'])

    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('vehicle_speed_delay_and_energy_consumption_bar_chart.pdf', format='pdf')
    plt.show()













def plot_PPO_rewards_beacon_cycle(rewards_1,ma_rewards_1,rewards_2,ma_rewards_2,rewards_3,ma_rewards_3,rewards_4,ma_rewards_4,rewards_5,ma_rewards_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    # plt.xticks([0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    # plt.yticks([-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')

    plt.plot(rewards_1,color="#EBD2D6")
    plt.plot(rewards_2,color="#B7D0EA")
    plt.plot(rewards_3,color="#CDE4E4")
    plt.plot(rewards_4,color="#F9F2C1")
    plt.plot(rewards_5,color="#DBD8E9")

    plt.plot(ma_rewards_1, label='beacon period=1',color="#C24976", linewidth=3)
    plt.plot(ma_rewards_2, label='beacon period=2',color="#407BD0", linewidth=3)
    plt.plot(ma_rewards_3, label='beacon period=3',color="#84C2AE", linewidth=3)
    plt.plot(ma_rewards_4, label='beacon period=4',color="#F4DEBB", linewidth=3)
    plt.plot(ma_rewards_5, label='beacon period=5',color="#B595BF", linewidth=3)


    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法

    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    legend =plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    legend.get_frame().set_alpha(0.5)
    plt.tight_layout()
    plt.savefig('PPO_rewards_beacon_period.pdf', format='pdf')
    plt.show()


def plot_PPO_completion_ratio_beacon_cycle(completion_ratio_1,ma_completion_ratio_1,completion_ratio_2,ma_completion_ratio_2,completion_ratio_3,ma_completion_ratio_3,completion_ratio_4,ma_completion_ratio_4,completion_ratio_5,ma_completion_ratio_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(completion_ratio_1,color="#EBD2D6")
    plt.plot(completion_ratio_2,color="#B7D0EA")
    plt.plot(completion_ratio_3,color="#CDE4E4")
    plt.plot(completion_ratio_4,color="#F9F2C1")
    plt.plot(completion_ratio_5,color="#DBD8E9")

    plt.plot(ma_completion_ratio_1, label='beacon period=1',color="#C24976", linewidth=3)
    plt.plot(ma_completion_ratio_2, label='beacon period=2',color="#407BD0", linewidth=3)
    plt.plot(ma_completion_ratio_3, label='beacon period=3',color="#84C2AE", linewidth=3)
    plt.plot(ma_completion_ratio_4, label='beacon period=4',color="#F4DEBB", linewidth=3)
    plt.plot(ma_completion_ratio_5, label='beacon period=5',color="#B595BF", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_completion_ratio_beacon_period.pdf', format='pdf')
    plt.show()




def plot_PPO_delay_beacon_cycle(delay_1,ma_delay_1,delay_2,ma_delay_2,delay_3,ma_delay_3,delay_4,ma_delay_4,delay_5,ma_delay_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('delay(s)', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( [0,400,800,1200,1600,2000],fontsize=22, fontname='Times New Roman')
    plt.plot(delay_1, color="#EBD2D6")
    plt.plot(delay_2, color="#B7D0EA")
    plt.plot(delay_3, color="#CDE4E4")
    plt.plot(delay_4, color="#F9F2C1")
    plt.plot(delay_5, color="#DBD8E9")

    plt.plot(ma_delay_1, label='beacon period=1', color="#C24976", linewidth=3)
    plt.plot(ma_delay_2, label='beacon period=2', color="#407BD0", linewidth=3)
    plt.plot(ma_delay_3, label='beacon period=3', color="#84C2AE", linewidth=3)
    plt.plot(ma_delay_4, label='beacon period=4', color="#F4DEBB", linewidth=3)
    plt.plot(ma_delay_5, label='beacon period=5', color="#B595BF", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_delay_beacon_period.pdf', format='pdf')
    plt.show()



def plot_PPO_energy_consumption_beacon_cycle(energy_consumption_1,ma_energy_consumption_1,energy_consumption_2,ma_energy_consumption_2,energy_consumption_3,ma_energy_consumption_3,energy_consumption_4,ma_energy_consumption_4,energy_consumption_5,ma_energy_consumption_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('energy consumption(J)', fontsize=26, fontname='Times New Roman')
    plt.xticks( [0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(energy_consumption_1, color="#EBD2D6")
    plt.plot(energy_consumption_2, color="#B7D0EA")
    plt.plot(energy_consumption_3, color="#CDE4E4")
    plt.plot(energy_consumption_4, color="#F9F2C1")
    plt.plot(energy_consumption_5, color="#DBD8E9")

    plt.plot(ma_energy_consumption_1, label='beacon period=1', color="#C24976", linewidth=3)
    plt.plot(ma_energy_consumption_2, label='beacon period=2', color="#407BD0", linewidth=3)
    plt.plot(ma_energy_consumption_3, label='beacon period=3', color="#84C2AE", linewidth=3)
    plt.plot(ma_energy_consumption_4, label='beacon period=4', color="#F4DEBB", linewidth=3)
    plt.plot(ma_energy_consumption_5, label='beacon period=5', color="#B595BF", linewidth=3)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_energy_consumption_beacon_period.pdf', format='pdf')
    plt.show()

def plot_beacon_cycle_delay_and_energy_consumption_bar_chart(beacon_cycle_delay_1, beacon_cycle_delay_2,
                                                                 beacon_cycle_delay_3, beacon_cycle_delay_4,
                                                                 beacon_cycle_delay_5,
                                                                 beacon_cycle_energy_consumption_1,
                                                                 beacon_cycle_energy_consumption_2,
                                                                 beacon_cycle_energy_consumption_3,
                                                                 beacon_cycle_energy_consumption_4,
                                                                 beacon_cycle_energy_consumption_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('beacon period', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average value', fontsize=26, fontname='Times New Roman')
    beacon_cycle_average_delay_1=sum(beacon_cycle_delay_1)/len(beacon_cycle_delay_1)
    beacon_cycle_average_delay_2 = sum(beacon_cycle_delay_2) / len(beacon_cycle_delay_2)
    beacon_cycle_average_delay_3 = sum(beacon_cycle_delay_3) / len(beacon_cycle_delay_3)
    beacon_cycle_average_delay_4 = sum(beacon_cycle_delay_4) / len(beacon_cycle_delay_4)
    beacon_cycle_average_delay_5 = sum(beacon_cycle_delay_5) / len(beacon_cycle_delay_5)

    beacon_cycle_average_energy_consumption_1 = sum(beacon_cycle_energy_consumption_1) / len(
        beacon_cycle_energy_consumption_1)
    beacon_cycle_average_energy_consumption_2 = sum(beacon_cycle_energy_consumption_2) / len(
        beacon_cycle_energy_consumption_2)
    beacon_cycle_average_energy_consumption_3 = sum(beacon_cycle_energy_consumption_3) / len(
        beacon_cycle_energy_consumption_3)
    beacon_cycle_average_energy_consumption_4 = sum(beacon_cycle_energy_consumption_4) / len(
        beacon_cycle_energy_consumption_4)
    beacon_cycle_average_energy_consumption_5 = sum(beacon_cycle_energy_consumption_5) / len(
        beacon_cycle_energy_consumption_5)

    delay=[beacon_cycle_average_delay_1,beacon_cycle_average_delay_2,beacon_cycle_average_delay_3,beacon_cycle_average_delay_4,beacon_cycle_average_delay_5]
    energy_consumption=[beacon_cycle_average_energy_consumption_1,beacon_cycle_average_energy_consumption_2,beacon_cycle_average_energy_consumption_3,beacon_cycle_average_energy_consumption_4,beacon_cycle_average_energy_consumption_5]

    Width = 0.2
    x1 = np.arange(len(delay))
    x2=[x + Width for x in x1]
    plt.bar(x1, delay,label='average delay',width=Width,color='#C3CEE4')
    plt.bar(x2, energy_consumption, label='average energy consumption',width=Width,color='#E8D6B6')

    plt.xticks([r + Width/2 for r in range(len(delay))], ['1','2','3','4','5'])

    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('beacon_period_delay_and_energy_consumption_bar_chart.pdf', format='pdf')
    plt.show()

def plot_beacon_cycle_delay_bar_chart(beacon_cycle_delay_1, beacon_cycle_delay_2,
                                                                 beacon_cycle_delay_3, beacon_cycle_delay_4,
                                                                 beacon_cycle_delay_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('beacon period', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average delay', fontsize=26, fontname='Times New Roman')
    beacon_cycle_average_delay_1=sum(beacon_cycle_delay_1)/len(beacon_cycle_delay_1)
    beacon_cycle_average_delay_2 = sum(beacon_cycle_delay_2) / len(beacon_cycle_delay_2)
    beacon_cycle_average_delay_3 = sum(beacon_cycle_delay_3) / len(beacon_cycle_delay_3)
    beacon_cycle_average_delay_4 = sum(beacon_cycle_delay_4) / len(beacon_cycle_delay_4)
    beacon_cycle_average_delay_5 = sum(beacon_cycle_delay_5) / len(beacon_cycle_delay_5)

    delay=[beacon_cycle_average_delay_1,beacon_cycle_average_delay_2,beacon_cycle_average_delay_3,beacon_cycle_average_delay_4,beacon_cycle_average_delay_5]
    x1=['1','2','3','4','5']
    plt.bar(x1, delay,color='#C3CEE4',width=0.4)




    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('beacon_period_delay_bar_chart.pdf', format='pdf')
    plt.show()


def plot_beacon_cycle_energy_consumption_bar_chart(beacon_cycle_energy_consumption_1,
                                                                 beacon_cycle_energy_consumption_2,
                                                                 beacon_cycle_energy_consumption_3,
                                                                 beacon_cycle_energy_consumption_4,
                                                                 beacon_cycle_energy_consumption_5):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('beacon period', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average energy consumption', fontsize=26, fontname='Times New Roman')


    beacon_cycle_average_energy_consumption_1 = sum(beacon_cycle_energy_consumption_1) / len(
        beacon_cycle_energy_consumption_1)
    beacon_cycle_average_energy_consumption_2 = sum(beacon_cycle_energy_consumption_2) / len(
        beacon_cycle_energy_consumption_2)
    beacon_cycle_average_energy_consumption_3 = sum(beacon_cycle_energy_consumption_3) / len(
        beacon_cycle_energy_consumption_3)
    beacon_cycle_average_energy_consumption_4 = sum(beacon_cycle_energy_consumption_4) / len(
        beacon_cycle_energy_consumption_4)
    beacon_cycle_average_energy_consumption_5 = sum(beacon_cycle_energy_consumption_5) / len(
        beacon_cycle_energy_consumption_5)

    energy_consumption=[beacon_cycle_average_energy_consumption_1,beacon_cycle_average_energy_consumption_2,beacon_cycle_average_energy_consumption_3,beacon_cycle_average_energy_consumption_4,beacon_cycle_average_energy_consumption_5]

    x2=['1','2','3','4','5']
    plt.bar(x2, energy_consumption, color='#C3CEE4',width=0.4)

    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('beacon_period_energy_consumption_bar_chart.pdf', format='pdf')
    plt.show()


# def plot_PPO_rewards_hop_number(rewards_1,ma_rewards_1,rewards_2,ma_rewards_2,rewards_3,ma_rewards_3,rewards_4,ma_rewards_4):
#     # sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
#     plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
#     # plt.title("Convergence graph for different number of processess", fontsize=14)
#     plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
#     plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
#     # plt.xticks([0,20,40,60,80,100,120,140,160,180,200],fontsize=22, fontname='Times New Roman')
#     # plt.yticks([-18000,-16000,-14000,-12000,-10000,-8000,-6000,-4000,-2000,0],fontsize=22, fontname='Times New Roman')
#
#     plt.plot(rewards_1,color="#EBD2D6")
#     plt.plot(rewards_2,color="#B7D0EA")
#     plt.plot(rewards_3,color="#CDE4E4")
#     plt.plot(rewards_4,color="#F9F2C1")
#
#
#     plt.plot(ma_rewards_1, label='hop number=1',color="#C24976", linewidth=3)
#     plt.plot(ma_rewards_2, label='hop number=2',color="#407BD0", linewidth=3)
#     plt.plot(ma_rewards_3, label='hop number=3',color="#84C2AE", linewidth=3)
#     plt.plot(ma_rewards_4, label='hop number=4',color="#F4DEBB", linewidth=3)
#
#
#
#     plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))  # 设置科学计数法
#
#     plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
#     legend =plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
#     legend.get_frame().set_alpha(0.5)
#     plt.tight_layout()
#     plt.savefig('PPO_rewards_hop_number.pdf', format='pdf')
#     plt.show()

def plot_hob_number_rewards(hob_number_1, hob_number_2,hob_number_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('hop count', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    x_datasize = ['1', '2', '3']
    hob_number_average_rewards_1=sum(hob_number_1)/len(hob_number_1)
    hob_number_average_rewards_2 = sum(hob_number_2) / len(hob_number_2)
    hob_number_average_rewards_3 = sum(hob_number_3) / len(hob_number_3)
    # hob_number_average_rewards_4 = sum(hob_number_4) / len(hob_number_4)
    delay = [hob_number_average_rewards_1, hob_number_average_rewards_2, hob_number_average_rewards_3]
    plt.plot(x_datasize, delay,  linestyle='-', marker='o', markersize=8 ,markeredgewidth=0,markerfacecolor='#C24976')



    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('hob_count_rewards_line_chart.pdf', format='pdf')
    plt.show()


def plot_hob_number_delay(hob_number_1, hob_number_2,hob_number_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('hop count', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average delay(s)', fontsize=26, fontname='Times New Roman')
    x_datasize = ['1', '2', '3']
    hob_number_average_delay_1=sum(hob_number_1)/len(hob_number_1)
    hob_number_average_delay_2 = sum(hob_number_2) / len(hob_number_2)
    hob_number_average_delay_3 = sum(hob_number_3) / len(hob_number_3)
    # hob_number_average_delay_4 = sum(hob_number_4) / len(hob_number_4)
    delay=[hob_number_average_delay_1,hob_number_average_delay_2,hob_number_average_delay_3]
    plt.plot(x_datasize, delay,  linestyle='-', marker='o', markersize=8 ,markeredgewidth=0,markerfacecolor='#C24976')



    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('hob_count_delay_line_chart.pdf', format='pdf')
    plt.show()


def plot_hob_number_energy_consumption(hob_number_1, hob_number_2,hob_number_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('hop count', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average energy consumption(J)', fontsize=26, fontname='Times New Roman')
    x_datasize = ['1', '2', '3']
    hob_number_average_energy_consumption_1=sum(hob_number_1)/len(hob_number_1)
    hob_number_average_energy_consumption_2 = sum(hob_number_2) / len(hob_number_2)
    hob_number_average_energy_consumption_3 = sum(hob_number_3) / len(hob_number_3)
    delay=[hob_number_average_energy_consumption_1,hob_number_average_energy_consumption_2,hob_number_average_energy_consumption_3]
    plt.plot(x_datasize, delay,  linestyle='-', marker='o', markersize=8 ,markeredgewidth=0,markerfacecolor='#C24976')



    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # 设置科学计数法
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    # plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('hob_count_energy_consumption_line_chart.pdf', format='pdf')
    plt.show()






#
#
# def plot_tasksize_four_dimensional(rewards1, delay1, energy_consumption1, rewards2, delay2, energy_consumption2, rewards3, delay3, energy_consumption3):
#     episodes = list(range(1, len(rewards1) + 1))
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     s1 = ax.scatter(episodes, delay1, energy_consumption1, c=rewards1, cmap='viridis', marker='o')
#     s2 = ax.scatter(episodes, delay2, energy_consumption2, c=rewards2, cmap='viridis', marker='D')
#     s3 = ax.scatter(episodes, delay3, energy_consumption3, c=rewards3, cmap='viridis', marker='s')
#     # 调整坐标轴
#     ax.set_xlabel('episodes')
#     ax.set_ylabel('delay')
#     ax.set_zlabel('energy consumption')
#     ax.view_init(elev=30, azim=15)
#     # ax.invert_xaxis()  # 反向x轴
#
#     # 添加颜色条和标签
#     cb = plt.colorbar(cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=-30000, vmax=0)), ax=ax)
#     cb.set_label('rewards')
#
#     plt.show()








# def plot_tasksize_four_dimensional(rewards1, delay1, energy_consumption1, rewards2, delay2, energy_consumption2, rewards3, delay3, energy_consumption3):
#     episodes = list(range(1, len(rewards1) + 1))
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 自定义颜色映射
#     cmap1 = LinearSegmentedColormap.from_list("custom_cmap1", ["red", "yellow"])
#     cmap2 = LinearSegmentedColormap.from_list("custom_cmap2", ["green", "blue"])
#     cmap3 = LinearSegmentedColormap.from_list("custom_cmap3", ["orange", "purple"])
#
#     # 绘制scatter，并使用不同的颜色映射
#     s1 = ax.scatter(episodes, delay1, energy_consumption1, c=rewards1, cmap=cmap1, marker='o')
#     s2 = ax.scatter(episodes, delay2, energy_consumption2, c=rewards2, cmap=cmap2, marker='D')
#     s3 = ax.scatter(episodes, delay3, energy_consumption3, c=rewards3, cmap=cmap3, marker='s')
#
#     # 调整坐标轴
#     ax.set_xlabel('episodes')
#     ax.set_ylabel('delay')
#     ax.set_zlabel('energy consumption')
#     ax.view_init(elev=30, azim=15)
#
#     # 添加颜色条和标签
#     cb1 = plt.colorbar(cm.ScalarMappable(cmap=cmap1), ax=ax)
#     cb1.set_label('rewards1')
#
#     cb2 = plt.colorbar(cm.ScalarMappable(cmap=cmap2), ax=ax)
#     cb2.set_label('rewards2')
#
#     cb3 = plt.colorbar(cm.ScalarMappable(cmap=cmap3), ax=ax)
#     cb3.set_label('rewards3')
#
#     plt.show()
#
#




#
# def plot_tasksize_four_dimensional(rewards1, delay1, energy_consumption1, rewards2, delay2, energy_consumption2, rewards3, delay3, energy_consumption3):
#     episodes = list(range(1, len(rewards1) + 1))
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # # 计算每个rewards列表的最小和最大值
#     # min_rewards1, max_rewards1 = min(rewards1), max(rewards1)
#     # min_rewards2, max_rewards2 = min(rewards2), max(rewards2)
#     # min_rewards3, max_rewards3 = min(rewards3), max(rewards3)
#
#     # # 创建归一化对象
#     # norm1 = Normalize(vmin=min_rewards1, vmax=max_rewards1)
#     # norm2 = Normalize(vmin=min_rewards2, vmax=max_rewards2)
#     # norm3 = Normalize(vmin=min_rewards3, vmax=max_rewards3)
#
#     # 绘制scatter，并使用不同的颜色映射
#     s1 = ax.scatter(episodes, delay1, energy_consumption1, c=rewards1, cmap='viridis', marker='o')
#     s2 = ax.scatter(episodes, delay2, energy_consumption2, c=rewards2, cmap='viridis', marker='D')
#     s3 = ax.scatter(episodes, delay3, energy_consumption3, c=rewards3, cmap='viridis', marker='s')
#
#     # 调整坐标轴
#     ax.set_xlabel('episodes')
#     ax.set_ylabel('delay')
#     ax.set_zlabel('energy consumption')
#     ax.view_init(elev=30, azim=15)
#
#     # 添加颜色条和标签
#     cb1 = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax)
#     cb1.set_label('rewards1')
#
#     cb2 = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax)
#     cb2.set_label('rewards2')
#
#     cb3 = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax)
#     cb3.set_label('rewards3')
#
#     plt.show()

#















def plot_tasksize_four_dimensional(rewards1, delay1, energy_consumption1, rewards2, delay2, energy_consumption2, rewards3, delay3, energy_consumption3):
    episodes = list(range(1, len(rewards1) + 1))

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 自定义颜色映射
    cmap1 = LinearSegmentedColormap.from_list("custom_cmap1", ["red", "yellow"])
    cmap2 = LinearSegmentedColormap.from_list("custom_cmap2", ["green", "blue"])
    cmap3 = LinearSegmentedColormap.from_list("custom_cmap3", ["orange", "purple"])

    # 绘制曲面，并为每个曲面设置不同的颜色映射
    surf1 = ax.plot_trisurf(episodes, delay1, energy_consumption1, cmap=cmap1, edgecolor='none', alpha=0.8)
    surf2 = ax.plot_trisurf(episodes, delay2, energy_consumption2, cmap=cmap2, edgecolor='none', alpha=0.8)
    surf3 = ax.plot_trisurf(episodes, delay3, energy_consumption3, cmap=cmap3, edgecolor='none', alpha=0.8)

    ax.view_init(elev=30, azim=0)

    # 设置颜色条和标签
    cb1 = fig.colorbar(surf1)
    cb1.set_label('rewards1')

    cb2 = fig.colorbar(surf2)
    cb2.set_label('rewards2')

    cb3 = fig.colorbar(surf3)
    cb3.set_label('rewards3')

    # 设置坐标轴标签
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Energy Consumption')

    plt.show()



def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """ 保存奖励 """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')







def make_dir(*paths):
    """ 创建文件夹 """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """ 删除目录下所有空文件夹 """
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")





def plot_Rewards(Rewards1,Rewards2,Rewards3,Rewards4):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    # plt.plot(PPO_tasksize_2_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_3_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_4_rewards,alpha=0.2)
    # plt.plot(PPO_tasksize_5_rewards,alpha=0.2)
    plt.plot(Rewards1, label='tasksize=1')
    plt.plot(Rewards2, label='tasksize=2')
    plt.plot(Rewards3, label='tasksize=3')
    plt.plot(Rewards4, label='tasksize=4')
    # plt.plot(Rewards5, label='tasksize=5')
    # plt.plot(PPO_tasksize_5_ma_rewards, label='lr=0.0001')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('PPO_rewards_lr.pdf', format='pdf')
    plt.show()