import torch
import torch.nn.functional as F
import bezier_renderer
import matplotlib.pyplot as plt
import os
import pydiffvg

def compute_cosine_similarity(feature):
    """计算特征向量间的余弦相似度矩阵"""
    norms = torch.norm(feature, p=2, dim=1, keepdim=True)
    feature_normalized = feature / norms
    cos_max = torch.mm(feature_normalized, feature_normalized.T)
    
    # 获取前三大相似度
    winner_1 = F.one_hot(torch.argmax(cos_max, dim=0), num_classes=16).T
    winner_2 = F.one_hot(torch.argmax(cos_max - cos_max * winner_1, dim=0), 
                        num_classes=16).T
    winner_3 = F.one_hot(torch.argmax(cos_max - cos_max * (winner_1 + winner_2), dim=0),
                        num_classes=16).T
                        
    reg_matrix = cos_max * (0.7*winner_1 + winner_2 * 0.2+ winner_3 * 0.1)
    return reg_matrix, cos_max