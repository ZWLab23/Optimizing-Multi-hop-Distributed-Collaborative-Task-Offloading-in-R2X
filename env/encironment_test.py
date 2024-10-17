
import numpy as np

# def calculate_node_values(node, graph, a1, a2, a3, p, discount, visited=None, memo=None):
#     if visited is None:
#         visited = set()
#     if memo is None:
#         memo = {}
#
#     # 若节点已经在之前计算过，则直接返回其价值
#     if node in memo:
#         return memo[node]
#
#     # 避免环的产生
#     if node in visited:
#         return (0, 0, 0)
#     visited.add(node)
#
#     # 获取节点的初始价值，假设所有未明确给出价值的节点价值为0
#     node_value1 = a1.get(node, 0)
#     node_value2 = a2.get(node, 0)
#     node_value3 = a3.get(node, 0)
#
#     # 遍历所有邻居
#     if node in graph:
#         for neighbor in graph[node]:
#             # 计算邻居贡献的价值
#             n_val1, n_val2, n_val3 = calculate_node_values(neighbor, graph, a1, a2, a3, p, discount, visited, memo)
#             prob = p.get((node, neighbor), 0)
#             contribution1 = prob * n_val1
#             contribution2 = prob * n_val2
#             contribution3 = prob * n_val3
#             node_value1 += discount * contribution1
#             node_value2 += discount * contribution2
#             node_value3 += discount * contribution3
#
#     # 将当前节点计算结果缓存并返回
#     memo[node] = (node_value1, node_value2, node_value3)
#     visited.remove(node)
#     return memo[node]
#
#
# # 示例
# graph = {
#     1: [2, 3, 4],
#     2: [1, 3, 5, 6],
#     3: [1, 2, 6],
#     4: [1],
#     5: [2],
#     6: [2, 3],
#     7:[]
# }
# a = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70}  # 给定每个节点的初值
# b={1: 100, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600, 7: 700}
# c={1: 1000, 2: 2000, 3: 3000, 4: 4000, 5: 5000, 6: 6000, 7: 7000}
# p = {
#     (1, 2): 0.1, (1, 3): 0.2, (1, 4): 0.7,
#     (2, 1): 0.1, (2, 3): 0.2, (2, 5): 0.3, (2, 6): 0.4,
#     (3, 1): 0.3, (3, 2): 0.3, (3, 6): 0.4,
#     (4, 1): 1,
#     (5, 2): 1,
#     (6, 2): 0.4, (6, 3): 0.6
# }  # 假设每个边缘的贡献
# discount = 0.9
#
# # 计算每个节点的价值
# node_values = {}
# for node in graph:
#     node_values[node] = calculate_node_values(node, graph, a, b,c,p, discount, visited=set(), memo={})
#
# for node, value in node_values.items():
#     print(f"Node {node} value: {value}")

# import math
#
# import math
#
# # 权重字典
# weight_dict= {
#     (0, 1): -5,
#     (0, 2): -6,
#     (1, 0): -3,
#     (1, 2): -4,
#     (2, 0): -7,
#     (2, 1): -8
# }
# source_weights_sum = {}
# for (source, _), weight in weight_dict.items():
#     source_weights_sum.setdefault(source, 0)
#     source_weights_sum[source] += math.exp(weight)
# selection_probability_dict={}
# # # 计算每个源节点对应的目标节点的概率
# for (source, target), weight in weight_dict.items():e
#     probability = math.exp(weight) / source_weights_sum[source]
#     selection_probability_dict[(source, target)] = probability
# print(selection_probability_dict)
#
# def coll(a,b,c):
#     class aa():
#         def __init__(self):
#             self.a=a
#             self.b=b
#             self.c=c
#         def aaa(self):
#             f=self.a+self.b+self.c
#             return f
#     j=aa()
#     return j.aaa()
# g=coll(1,2,3)
# print(g)
#

#
# def inverse_composite_action(mapped_value):
#
#     # 计算 y 值，因为 y 的范围是 1 到 10，所以直接用整除运算得到 y
#     y = mapped_value // 10 + 1
#     # 计算 x 值，x 是剩余的部分
#     x = mapped_value % 10 + 1
#     return x, y
# for i in range(100):
#     a=inverse_composite_action(i)
#     print(a)
# def isin():
# 动作映射

# 定义映射函数，将 (x, y, z) 映射为整数
# def composite_action(x, y, z):
#     return (x - 1) + (y - 1) * 5 + (z - 1) * 5 * 5+1
# print(composite_action(4,4,4))
#
# def inverse_composite_action(action):
#     # 计算 z
#     r = 3 + 2
#     z = action // (r * r)
#     # 计算 y
#     y = (action // r) % r
#     # 计算 x
#     x = action % r
#     # 将结果加上1，使得范围变为1到r
#     x += 1
#     y += 1
#     z += 1
#     action_list = [x - 1, y - 1, z - 1]
#     return action_list
# print(inverse_composite_action(93))

#
# def action_conversion(action):
#     # 计算 y 值，因为 y 的范围是 1 到 10，所以直接用整除运算得到 y
#     y = action // 10
#     # 计算 x 值，x 是剩余的部分
#     x = action % 10
#     x += 1
#     y += 1
#     return x, y - 1
# print(action_conversion(41))

#
#
# #获取任务属性的类：获取任务信息大小，计算能力：每bit所需转数，任务延迟约束
# class Function(object):
#     """任务属性及其操作"""
#     #就是一个任务的三元数组
#
#     def __init__(
#             self,
#             Function_task_datasize: float,
#             Function_task_computing_resource: float,
#             Function_task_delay: int
#
#     ) -> None:
#         self._Function_task_datasize = Function_task_datasize   #感知任务的大小
#         self._Function_task_computing_resource = Function_task_computing_resource   #感知任务每bit的所需计算资源
#         self._Function_task_delay = Function_task_delay #感知任务的延迟
#     def get_task_datasize(self) -> float:
#         return float(self._Function_task_datasize)
#
#     def get_task_computing_resource(self) -> float:
#         return float(self._Function_task_computing_resource)
#
#     def get_task_delay(self) -> float:
#         return float(self._Function_task_delay)

#
#
# def _function_generator() :
#     """ 产生我们关注的任务 """
#     new_function = []
#
#     for i in range(3):
#         # np.random.seed(self.seed + i)
#
#         Function_task_datasize = np.random.uniform(1,
#                                                    5)
#         Function_task_delay = int(
#             np.random.uniform(10, 20))
#
#         function = Function(Function_task_datasize, 300,
#                             Function_task_delay)
#         new_function.append(function)
#     return new_function
#
# def create_function_task():
#     new_function = _function_generator()
#     new_task_list = []
#     for i in range(3):
#         task_list = []
#         task_list.append(i)
#         task_list.append(new_function[i].get_task_datasize())
#         task_list.append(new_function[i].get_task_computing_resource())
#         task_list.append(new_function[i].get_task_delay())
#         new_task_list.append(task_list)
#     return new_task_list.copy()
#
# print(create_function_task())
# print(create_function_task())
# print(create_function_task())
#
# all_vehicle_number=[[1,4],(9,4)]
# neighbors_dict={(10,4):2}
# is_not_execute=0
# check_last_node=10
# for m in range(len(all_vehicle_number)):
#     if all_vehicle_number[m][1] not in neighbors_dict[check_last_node] and all_vehicle_number[m][1] != check_last_node:
#         is_not_execute = is_not_execute + 1
#         check_last_node = all_vehicle_number[m][1]
#     #
# for n in range(len(all_vehicle_number)):
#     if all_vehicle_number[n][1] == check_last_node_1 and all_vehicle_number[n][0] != 10:
#         is_not_execute = is_not_execute + 1
#         check_last_node_1 = all_vehicle_number[n][1]
#
#     # if all_vehicle_number[-1][0] !=self.config.vehicle_number:
#     #     reward = self.config.RSUDRL_punishment
#     #     reward_list.append(reward)
#     #     success_task.append(0)
# if is_not_execute != 0:
#     reward = 200
#     reward_list.append(reward)
#     success_task.append(0)
# import tensorflow as tf
#
#
# def predict_speed( history_speed, n_steps):
#     x, y = list(), list()
#     for i in range(len(history_speed)):
#         end_ix = i + n_steps
#         if end_ix > len(history_speed) - 1:
#             break
#         seq_x, seq_y = history_speed[i:end_ix], history_speed[end_ix]
#         x.append(seq_x)
#         y.append(seq_y)
#     X = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     print(X,y)
#     n_features = 1
#     X = tf.reshape(X, (X.shape[0], X.shape[1], n_features))
#
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
#         tf.keras.layers.Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=200, verbose=0)
#
#     x_input = tf.convert_to_tensor(history_speed[-n_steps:], dtype=tf.float32)
#     x_input = tf.reshape(x_input, (1, n_steps, n_features))
#     yhat = model(x_input, training=False)
#     yhat = int(yhat[0, 0])
#
#     return yhat
#
# a=[10,20,30,40,50,60,70,80,90]
# b=predict_speed( a, 3)
# print(b)


# # 权重字典
# weight_dict= {
#     (0, 1): (-5,1),
#     (0, 2): -6,
#     (1, 0): -3,
#     (1, 2): -4,
#     (2, 0): -7,
#     (2, 1): -8
# }
# print(len(weight_dict[(0,1)]))
def decomposition_action( action):
    # 计算 y 值，因为 y 的范围是 1 到 10，所以直接用整除运算得到 y
    y = action // 9
    # 计算 x 值，x 是剩余的部分
    x = action % 9
    x += 1
    y += 1
    return x - 1, y - 1
print(decomposition_action(80))