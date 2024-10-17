import numpy as np
from env.config_contrast import VehicularEnvConfig
class Greedy(object):
    """ 贪心算法实现思路 """


    def __init__(self):
        self.config=VehicularEnvConfig()
        pass

    def choose_action(self, state,neighbor_vehicle_list,new_task_list) -> int:
        """ 根据任务队列选择合适的卸载节点 """

        State=state
        function_size=new_task_list[1]
        #
        # min_index=neighbor_vehicle_list[0]
        # for i in range(len(neighbor_vehicle_list)):
        #     if State[neighbor_vehicle_list[i]]<State[min_index]:
        #         min_index=neighbor_vehicle_list[i]
        # for j in range(self.config.rsu_number):
        #     if State[j] < State[min_index]:
        #         min_index = j

        min_index=np.argmin(State)
        action_list=min_index


        return action_list
