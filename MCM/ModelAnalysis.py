"""
    coding      : UTF-8
    Environment : Python 3.6
    Author      : Benjamin142857
    Data        : 9/2/2019
    Remark      : ModelAnalysis - 分析模型
"""
import numpy as np


def ahp(X):
    """
    层次分析法（Analytic Hierarchy Process）
    :param X: <np.matrix> 方阵
    :return ret_dict: <dict> 结果字典: {
                        'CR': 一致性比例值,
                        'D': 特征值向量,
                        'D_max': 最大特征值
                        'V': 特征向量矩阵
                        'w': 特征权值列表（最大特征向量归一化结果）
                    }
    """

    # 0. init
    RI = [0, 0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]    # 一致性检验RI值表
    n = X.shape[0]   # 特征数
    ret_dict = {}

    # 1. 特征分解
    [D, V] = np.linalg.eig(X)
    D = D.astype('float')
    V = V.astype('float')
    ret_dict['D'] = D
    ret_dict['V'] = V

    # 2. 求最大特征值及其索引
    D_max = max(D)
    D_maxi = list(D).index(max(D))
    ret_dict['D_max'] = D_max

    # 3. 求CI与CR
    CI = (D_max-n) / (n-1)
    CR = CI / RI[n]
    ret_dict['CR'] = CR

    # 4. 求权值列表
    w = V[:, D_maxi]/sum(V[:, D_maxi])
    ret_dict['w'] = w

    return ret_dict


if __name__ == '__main__':
    X = np.matrix([
        [1, 2 / 1, 5 / 1, 3 / 1],
        [1 / 2, 1 / 1, 3 / 1, 1 / 2],
        [1 / 5, 1 / 3, 1 / 1, 1 / 4],
        [1 / 3, 2 / 1, 4 / 1, 1 / 1],
    ])

    res_dct = ahp(X)

    for k, v in res_dct.items():
        print(k, v)
