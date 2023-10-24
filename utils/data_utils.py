# -*- coding: utf-8 -*-

import numbers
import numpy as np
from functools import singledispatch


@singledispatch
def uniform_sampling(minmax_range, data_n):
    _temp_data = np.random.random_sample((data_n, 1))
    _temp_data = (minmax_range[1] - minmax_range[0]) * _temp_data + minmax_range[0]
    return _temp_data


@uniform_sampling.register(numbers.Real)
def _(_value, data_n):
    _temp_data = np.ones((data_n, 1)) * _value
    return _temp_data


@uniform_sampling.register(tuple)
def _(_value, data_n):
    random_mask = np.random.randn(data_n, 1) > 0
    _temp_data = random_mask * _value[0] + (1-random_mask) * _value[1]
    return _temp_data


def split_data(data, t_split_dict, t_dim_index):
    # 检查多个分段是否相同
    # 如果多个分段完全相同则随机采样
    temp_list = list()
    random_split = True
    for split_key, t_list in t_split_dict.items():
        if len(temp_list) == 0:
            temp_list = t_list
        else:
            if temp_list[0] != t_list[0] or temp_list[1] != t_list[1]:
                random_split = False
                break

    split_result = dict()
    if random_split:
        random_index_list = np.random.permutation(len(data))
        sample_each_group = int(len(random_index_list) / len(t_split_dict))
        for i, split_key in enumerate(t_split_dict.keys()):
            _temp_data_indices = random_index_list[i * sample_each_group:(i + 1) * sample_each_group]
            split_result[split_key] = data[_temp_data_indices, :]
    else:
        for split_key, t_list in t_split_dict.items():
            _temp_data_indices = np.argwhere((data[:, t_dim_index] > t_list[0]) &
                                             (data[:, t_dim_index] < t_list[1])).reshape(-1)

            split_result[split_key] = data[_temp_data_indices, :]

    return split_result
