import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import pydiffvg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import display_png
import sketch_utils as utils
import seaborn as sns
#计算余弦相似度
def cos_sim_matrix(feature):
    #feature:matrix[num_feature,vec_dim] eg.16,8
    # 计算 L2 范数
    norms = torch.norm(feature, p=2, dim=1, keepdim=True)  # [16, 1]
    # 标准化
    feature_normalized = feature / norms  # [16, 8]
    # 计算两两余弦相似度
    cos_matrix = torch.mm(feature_normalized, feature_normalized.T)  # [16, 16]
    winner_1 = F.one_hot(torch.argmax(cos_max, dim=0), num_classes=16).T
    winner_2 = F.one_hot(torch.argmax(cos_max - cos_max * winner_1, dim=0), num_classes=16).T
    winner_3 = F.one_hot(torch.argmax(cos_max - cos_max * (winner_1 + winner_2), dim=0), num_classes=16).T
    # print(f"w2转置后给w3得出的第三大的值{torch.argmax(cos_max - cos_max * (winner_1 + winner_2_nex), dim=0)}")
    reg_matrix = cos_max * (winner_1 + winner_2 * 0.5 + winner_3 * 0.2)
    return reg_matrix
def Cos_matrix_2(feature):
    torch.set_printoptions(precision=15)
    cosine_sim_matrix = torch.zeros((feature.shape[0], feature.shape[0]))
    # 计算行与行之间的余弦相似度
    for i in range(feature.shape[0]):
        for j in range(i, feature.shape[0]):  # 只计算上三角部分
            # 计算点积
            dot_product = torch.dot(feature[i], feature[j])
            # 计算 L2 范数
            norm_i = torch.sqrt(torch.sum(feature[i] ** 2))
            norm_j = torch.sqrt(torch.sum(feature[j] ** 2))
            # 计算余弦相似度
            cosine_similarity = dot_product / (norm_i * norm_j)
            # 将计算结果存入相似度矩阵
            cosine_sim_matrix[i, j] = cosine_similarity
            if i != j:
                cosine_sim_matrix[j, i] = cosine_similarity  # 利用对称性填充矩阵的对称部分
    print(f"the cosine_sim_matrix {cosine_sim_matrix.shape}")
    winner_1 = F.one_hot(torch.argmax(cosine_sim_matrix, dim=0), num_classes=16)
    # print(f"the winner_1 is {winner_1}")
    winner_2 = F.one_hot(torch.argmax(cosine_sim_matrix - cosine_sim_matrix * winner_1, dim=0), num_classes=16).T
    # print(f"the winner_2 is {winner_2}")
    winner_3 = F.one_hot(torch.argmax(cosine_sim_matrix - cosine_sim_matrix * (winner_1 + winner_2), dim=0), num_classes=16).T
    # print(f"w2转置后给w3得出的第三大的值{torch.argmax(cos_max - cos_max * (winner_1 + winner_2_nex), dim=0)}")
    reg_matrix = cosine_sim_matrix * (winner_1 + winner_2 * 0.5 + winner_3 * 0.2)
    return reg_matrix
if __name__ == '__main__':
    # 原始的 Tensor 数据
    coordinates = torch.tensor([[[111.8322, 71.3526],
                                 [138.0330, 71.9311],
                                 [141.0354, 77.7462],
                                 [132.2314, 76.8853]],

                                [[98.8166, 98.0967],
                                 [84.8215, 94.1833],
                                 [93.9764, 99.3244],
                                 [102.8471, 93.6680]],

                                [[136.0715, 78.3043],
                                 [162.4782, 103.8046],
                                 [167.8317, 90.0466],
                                 [163.0202, 95.2653]],

                                [[40.3431, 54.4434],
                                 [43.7558, 82.6741],
                                 [52.9506, 80.7047],
                                 [59.9959, 94.8020]],

                                [[111.5894, 66.4357],
                                 [103.3541, 56.6692],
                                 [103.3598, 46.4232],
                                 [86.6925, 37.1566]],

                                [[56.7280, 92.3222],
                                 [66.0501, 99.3159],
                                 [64.9003, 94.4310],
                                 [43.0570, 82.1070]],

                                [[162.7955, 94.2949],
                                 [179.7129, 83.9759],
                                 [175.1923, 67.7477],
                                 [176.3078, 57.1743]],

                                [[102.9343, 96.4623],
                                 [105.7106, 115.2663],
                                 [113.9359, 106.8916],
                                 [117.5628, 134.3228]],

                                [[42.8805, 120.9556],
                                 [21.5091, 175.3311],
                                 [25.5458, 151.8602],
                                 [47.4889, 196.7332]],

                                [[110.6335, 75.6564],
                                 [113.2407, 57.5218],
                                 [109.8890, 58.7882],
                                 [103.3897, 52.5767]],

                                [[59.6197, 94.8582],
                                 [81.0271, 115.6852],
                                 [63.1067, 106.4087],
                                 [106.4332, 117.5909]],

                                [[25.2464, 89.4446],
                                 [14.0712, 79.9732],
                                 [19.9463, 82.2277],
                                 [19.4858, 83.6646]],

                                [[168.1984, 100.1352],
                                 [192.9689, 88.6479],
                                 [170.5703, 51.7429],
                                 [203.9741, 60.0048]],

                                [[106.6212, 119.7323],
                                 [109.0889, 148.5027],
                                 [100.3025, 162.9088],
                                 [111.6081, 153.8226]],

                                [[175.6760, 54.9666],
                                 [184.0321, 12.3017],
                                 [189.4098, 43.7307],
                                 [200.8110, 42.0317]],

                                [[201.8784, 55.4477],
                                 [213.0451, 41.5494],
                                 [186.6398, 54.7259],
                                 [204.0457, 51.2076]]])

