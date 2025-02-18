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
                        
    reg_matrix = cos_max * (winner_1 + winner_2 * 0.5 + winner_3 * 0.2)
    return reg_matrix, cos_max

def check_gradients(model, threshold=1e-5):
    """检查模型参数的梯度
    
    Args:
        model: PyTorch模型
        threshold: 梯度阈值,小于此值视为梯度消失
        
    Returns:
        dict: 包含梯度统计信息的字典
    """
    stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'params_without_grad': 0,
        'max_grad': 0,
        'min_grad': float('inf'),
        'mean_grad': 0
    }
    
    grads = []
    
    for name, param in model.named_parameters():
        stats['total_params'] += 1
        
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grads.append(grad_norm)
            
            stats['params_with_grad'] += 1
            stats['max_grad'] = max(stats['max_grad'], grad_norm)
            stats['min_grad'] = min(stats['min_grad'], grad_norm)
            
            if grad_norm < threshold:
                print(f'警告: 参数 {name} 的梯度接近于0: {grad_norm:.6f}')
        else:
            stats['params_without_grad'] += 1
            print(f'警告: 参数 {name} 没有梯度')
            
    if grads:
        stats['mean_grad'] = sum(grads) / len(grads)
        
    return stats
