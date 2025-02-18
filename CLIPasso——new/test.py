import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)
        self.fc3 = nn.Linear(4, 4)  # 多余的FC
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
def cos_Adjmatrix(feature):
    #feature:matrix[num_feature,vec_dim] eg.16,8
    # 计算 L2 范数
    norms = torch.norm(feature, p=2, dim=1, keepdim=True)  # [16, 1]
    # 标准化
    feature_normalized = feature / norms  # [16, 8]
    # 计算两两余弦相似度
    cos_max = torch.mm(feature_normalized, feature_normalized.T)  # [16, 16]
    # for i in range(cos_max.shape[0]):
        # print(f"cos_max: {i,cos_max[i]}")
    winner_1 = F.one_hot(torch.argmax(cos_max, dim=0), num_classes=16).T
    # print(f"the winner_1 is {winner_1}")
    winner_2 = F.one_hot(torch.argmax(cos_max - cos_max * winner_1, dim=0), num_classes=16).T
    # print(f"the winner_2 is {winner_2}")
    winner_3 = F.one_hot(torch.argmax(cos_max - cos_max * (winner_1 + winner_2), dim=0), num_classes=16).T
    # print(f"w2转置后给w3得出的第三大的值{torch.argmax(cos_max - cos_max * (winner_1 + winner_2_nex), dim=0)}")
    reg_matrix = cos_max * (0.7*winner_1 + winner_2 * 0.2 + winner_3 * 0.1)
    return reg_matrix,cos_max

def cs_heatmap(reg_matrix):
    # reg_matrix:matrix
    # save_dir = os.path.join(folder_path, f"{epoch}_{name}.png")
    # 2. 将余弦相似度矩阵转换为 numpy 数组
    cos_sim_matrix_np = reg_matrix.clone().detach().cpu().numpy()
    # 在每个单元格中添加文本标注
    for i in range(cos_sim_matrix_np.shape[0]):
        for j in range(cos_sim_matrix_np.shape[1]):
            if cos_sim_matrix_np[i, j] > 0:
                # 在矩阵中添加对应的像素值
                plt.text(j, i, f'{cos_sim_matrix_np[i, j]:.2f}', ha='center', va='center', color='black', fontsize=10)
    # 使用 imshow 绘制热力图
    plt.imshow(cos_sim_matrix_np, cmap='autumn', interpolation='nearest')
    # 显示颜色对应条
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig("save_dir.png")
    plt.clf()


if __name__ == "__main__":

    test_tensor=torch.tensor([[0.6333, 0.4577, 0.3132, 0.2292, 0.4919, 0.1120, 0.6245, 0.4230],
        [0.3029, 0.2097, 0.2485, 0.4133, 0.0948, 0.4712, 0.2030, 0.8542],
        [0.8517, 0.2343, 0.8094, 0.4565, 0.7051, 0.5508, 0.5503, 0.3280],
        [0.5056, 0.5546, 0.2582, 0.5666, 0.1359, 0.3282, 0.2089, 0.5475],
        [0.7180, 0.4132, 0.5449, 0.4612, 0.4655, 0.5222, 0.4862, 0.4970],
        [0.5258, 0.3233, 0.5924, 0.6235, 0.5342, 0.5441, 0.4216, 0.3878],
        [0.7786, 0.2504, 0.7140, 0.5584, 0.6544, 0.3633, 0.5748, 0.4574],
        [0.4862, 0.8273, 0.4159, 0.4793, 0.5957, 0.2834, 0.4609, 0.8394],
        [0.5103, 0.4847, 0.5864, 0.3318, 0.6463, 0.4244, 0.5554, 0.3436],
        [0.5967, 0.4536, 0.6044, 0.3614, 0.6713, 0.4480, 0.7359, 0.3788],
        [0.3340, 0.4944, 0.5556, 0.4514, 0.5507, 0.5198, 0.3541, 0.4549],
        [0.3979, 0.3154, 0.4494, 0.4491, 0.4163, 0.5270, 0.4221, 0.3239],
        [0.4820, 0.5928, 0.4601, 0.5251, 0.5257, 0.6851, 0.4243, 0.4546],
        [0.1532, 0.3640, 0.3295, 0.4793, 0.5439, 0.4446, 0.3643, 0.2070],
        [0.5064, 0.5462, 0.6433, 0.4088, 0.6042, 0.2155, 0.3946, 0.3543],
        [0.4886, 0.4045, 0.3413, 0.3068, 0.4775, 0.5289, 0.4619, 0.4538]])


    print(test_tensor.shape)
    _,cos_test=cos_Adjmatrix(test_tensor)
    cs_heatmap(cos_test)
    print(cos_test)
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(test_tensor[13], test_tensor[14],dim=0)
    print("Cosine similarity:", cosine_similarity.item())