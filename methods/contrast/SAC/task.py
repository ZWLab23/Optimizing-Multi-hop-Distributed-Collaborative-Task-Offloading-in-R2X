import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
# print(curr_path)

# sys.path.append(parent_path)  # 添加路径到系统路径
parent_path_1 = os.path.dirname(parent_path)
sys.path.append(parent_path_1)
# print(parent_path)
import torch
from env.environment_contrast_Lyapunov import LyapunovModel
from methods.contrast.SAC.sac import SAC
from env.config_contrast import VehicularEnvConfig
import numpy as np
import dill as pickle  # 用dill代替pickle,保存参数的
from env.utils import plot_rewards,  save_results_1, plot_completion_rate, plot_delay, plot_energy_consumption,plot_backlogs
#  --------------------------------基础准备--------------------------------  #
algo_name = "SAC"  # 算法名称
env_name = "LyapunovModel"  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
import datetime

#  --------------------------------训练与测试--------------------------------  #
class SACConfig:
    """ 算法超参数 """

    def __init__(self) -> None:
        # 准备工作
        self.algo_name = algo_name
        self.env_name = env_name
        self.device = device
        # 训练设置
        self.train_eps = 300
        self.test_eps = 20
        self.max_steps = 1000  # 每回合的最大步数
        # 网络参数
        self.hidden_dim = 256
        self.value_lr = 0.0002
        self.soft_q_lr = 0.0002
        self.policy_lr = 0.0002
        self.mean_lambda = 1e-4
        self.std_lambda = 1e-2
        self.z_lambda = 0.0
        self.soft_tau = 1e-2  # 目标网络软更新参数
        # 折扣因子
        self.gamma = 0.99
        # 经验池
        self.capacity = 1000000
        self.batch_size = 128

        self.stability_tag = "a"
        self.flow_tag = "flow"
        ################################################################################
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
        self.result_path=curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'
        self.model_path=curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/models/'
        self.save_fig=True
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU


def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = LyapunovModel(cfg.stability_tag,cfg.flow_tag)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    n_actions = env.action_space.shape[0]  # 动作维度
    agent = SAC(n_states, n_actions, cfg) # 创建智能体
    # if hasattr(env, 'reset'):  # 检查环境是否具有 reset 方法
    #     if cfg.seed != 0:  # 设置随机种子
    #         torch.manual_seed(cfg.seed)
    #         env.reset(seed=cfg.seed)
    #         np.random.seed(cfg.seed)

    return env, agent


def train(cfg, env, agent):
    print("Start training!")
    print(f"Env:{cfg.env_name}, Algo:{cfg.algo_name}, Device:{cfg.device}")

    rewards_plot = []  # 记录所有回合的奖励画图
    ma_rewards_plot = []  # 记录所有回合的滑动平均奖励画图
    train_episodes = []  # 记录所有回合数，用于画图
    delay_plot = []
    ma_delay_plot = []
    backlogs_plot = []
    ma_backlogs_plot = []
    completion_rate_plot=[]
    ma_completion_rate_plot = []



    for i_ep in range(cfg.train_eps):
        train_episodes.append(i_ep)
        ep_completed = 0
        ep_backlog = 0
        ep_reward = 0  # 记录一回合内的奖励
        ep_delay = 0
        ep_vehicle_queue_length = 0
        ep_rsu_queue_length = 0
        ep_queue_length = 0
        ep_vehicle_y = 0
        ep_rsu_y = 0
        ep_y = 0
        state = env.reset()  # 重置环境，返回初始状态
        for i_step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward,success_task, backlog, delay, done, queue_v, y_v, queue_r, y_r, queue, y = env.step(action)
            ep_completed =ep_completed +success_task
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            ep_backlog += backlog
            ep_delay += delay
            ep_vehicle_queue_length += queue_v
            ep_rsu_queue_length += queue_r
            ep_queue_length += queue
            ep_vehicle_y += y_v
            ep_rsu_y += y_r
            ep_y += y

            if done:
                break
        rewards_plot.append(ep_reward)
        delay_plot.append(ep_delay)
        backlogs_plot.append(ep_backlog)
        completion_rate=ep_completed/(env.config.time_slot_end + 1)
        completion_rate_plot.append(completion_rate)

        print("#  episode :{}, rewards : {}, delay : {}, backlogs : {}, complete : {}"
              .format(i_ep+1,ep_reward,ep_delay,ep_backlog,completion_rate))

        # 检查目录是否存在，如果不存在，则创建它
        if not os.path.exists(cfg.result_path):
            os.makedirs(cfg.result_path)

        # 打开一个TXT文件，如果文件不存在则创建它，并以追加模式打开
        with open(cfg.result_path+'experimental_result.txt', 'a') as file:
            # 写入数据
            file.write("#  episode :{}, rewards : {}, delay : {}, backlogs : {}, complete : {}\n".format(i_ep+1,ep_reward,ep_delay,ep_backlog,completion_rate))

        if ma_rewards_plot:
            ma_rewards_plot.append(0.9 * ma_rewards_plot[-1] + 0.1 * ep_reward)
        else:
            ma_rewards_plot.append(ep_reward)

        if ma_delay_plot:
            ma_delay_plot.append(0.9 * ma_delay_plot[-1] + 0.1 * ep_delay)
        else:
            ma_delay_plot.append(ep_delay)

        if ma_backlogs_plot:
            ma_backlogs_plot.append(0.9 * ma_backlogs_plot[-1] + 0.1 * ep_backlog)
        else:
            ma_backlogs_plot.append(ep_delay)

        if ma_completion_rate_plot:
            ma_completion_rate_plot.append(0.9 * ma_completion_rate_plot[-1] + 0.1 * completion_rate)
        else:
            ma_completion_rate_plot.append(completion_rate)

    print("Finish training!")
    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    with open(cfg.model_path+'SACDRL_parameters.pkl', 'wb') as f:
        pickle.dump(agent, f)

    res_dic_rewards = {'rewards': rewards_plot, 'ma_rewards': ma_rewards_plot}
    res_dic_delay = {'delay': delay_plot, 'ma_delay': ma_delay_plot}
    res_dic_backlogs = {'backlogs': backlogs_plot, 'ma_backlogs': ma_backlogs_plot}
    res_dic_completion_rate = {'completion_rate': completion_rate_plot,
                               'ma_completion_rate': ma_completion_rate_plot}

    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_delay, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_backlogs, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='train',
                   path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_delay(res_dic_delay['delay'], res_dic_delay['ma_delay'], cfg, tag="train")
    plot_backlogs(res_dic_backlogs['backlogs'], res_dic_backlogs['ma_backlogs'], cfg, tag="train")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'],
                         cfg, tag="train")
    env.close()




if __name__ == "__main__":
    cfg = SACConfig()
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