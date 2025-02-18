import array
import pydiffvg
from PIL import Image,ImageOps
from typing import Optional, Tuple, Any
import os, random, pdb
import torch, numpy as np
from matplotlib.transforms import Affine2D
from numpy import ndarray, dtype
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


import struct
import sys
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import functional as F
import torch.nn.functional as F2

class BezierRenderer:
    """贝塞尔曲线渲染器类"""

    def __init__(self, canvas_width=224, canvas_height=224):
        """初始化渲染器

        Args:
            canvas_width: 画布宽度,默认224
            canvas_height: 画布高度,默认224
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def render_img_raw(self, sketch):
        """将贝塞尔曲线控制点转换为渲染图像，处理一张图片，不放大

        Args:
            sketch: 贝塞尔曲线控制点列表,每条曲线包含4个控制点(4x2)

        Returns:
            gray_img: 渲染后的灰度图像,形状为(H,W,1)
        """
        shapes = []
        shape_groups = []

        # 遍历每条曲线的控制点
        for i, stroke in enumerate(sketch):
            num = stroke.shape[0]
            # 每条曲线使用2个控制点
            num_control_points = torch.tensor([2], dtype=torch.int32)
            points = stroke.contiguous()

            # 创建Path对象表示一条曲线
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 is_closed=False,
                                 stroke_width=torch.tensor(1))
            shapes.append(path)

            # 创建ShapeGroup对象设置曲线样式(黑色描边,无填充)
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i], dtype=torch.int32),
                fill_color=None,
                stroke_color=torch.tensor([0, 0, 0, 1.0], dtype=torch.float32)
            )
            shape_groups.append(shape_group)

        # 序列化场景参数
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups)

        # 创建白色背景
        background_image = torch.ones(self.canvas_height, self.canvas_width, 4)
        background_image[:, :, 0:3] = 1.0  # RGB通道设为1(白色)
        background_image[:, :, 3] = 1.0  # Alpha通道设为1(不透明)

        # 渲染图像
        render = pydiffvg.RenderFunction.apply
        img = render(self.canvas_width, self.canvas_height, 2, 2, 0,
                     background_image, *scene_args)

        # 转换为灰度图
        img = img[:, :, :3]  # 去掉alpha通道
        gray_img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        gray_img = gray_img.unsqueeze(2)
        return gray_img

    def render_bezier_single(self, control_points):
        """渲染单条贝塞尔曲线

        Args:
            control_points: 单条曲线的控制点(4x2)

        Returns:
            gray_img: 渲染后的灰度图像,形状为(H,W,1)
        """
        num_points = 2
        list_control_points = control_points.contiguous()
        path = pydiffvg.Path(
            num_control_points=torch.tensor([num_points]),
            points=list_control_points,
            stroke_width=torch.tensor(1),
            is_closed=False
        )
        shapes = [path]

        shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        )
        shape_groups = [shape_group]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups)

        background_image = torch.ones(self.canvas_height, self.canvas_width, 4)
        background_image[:, :, 0:3] = 1.0
        background_image[:, :, 3] = 1.0

        render = pydiffvg.RenderFunction.apply
        img = render(self.canvas_width, self.canvas_height, 2, 2, 0,
                     background_image, *scene_args)

        gray_img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        gray_img = gray_img.unsqueeze(2)

        return gray_img

    def render_beziers(self, control_points):
        """渲染控制点生成贝塞尔曲线集

        Args:
            control_points: 所有曲线的控制点列表

        Returns:
            torch.Tensor: 渲染后的图像列表
        """
        img_beziers = []
        for points in control_points:
            img = self.render_bezier_single(points.clone())
            img_beziers.append(img.unsqueeze(0))
        return torch.cat(img_beziers, dim=0)


    def _get_nonzero_indices(self, mask):
        """获取mask中非零区域的边界框索引

        Args:
            mask: 输入mask图像

        Returns:
            tuple: (y_min, y_max, x_min, x_max)或None
        """
        mask_nonzero = mask.view(-1)
        indices = mask_nonzero.nonzero().squeeze()

        if indices.numel() > 0:
            rows = indices // mask.shape[1]
            cols = indices % mask.shape[1]
            return rows.min(), rows.max(), cols.min(), cols.max()
        return None

    def mask_img(self, control_points):
        """生成每条曲线的mask图像

        Args:
            control_points: 所有曲线的控制点

        Returns:
            torch.Tensor: 每条曲线对应的mask图像,CHW格式,大小64x64
        """
        img_rare = self.render_img_raw(control_points)
        img_beziers = self.render_beziers(control_points)
        # print(img_rare.shape)
        img_masked = []
        for img_mask in img_beziers:
            # 生成二值mask
            binary_mask = (img_mask > 0).float() * 255
            reverse_mask = 255 - binary_mask

            # 获取非零区域的边界框
            mask_indices = self._get_nonzero_indices(reverse_mask)
            if mask_indices is not None:
                y_min, y_max, x_min, x_max = mask_indices

                # 生成矩形mask并裁剪图像
                rect_mask = torch.zeros_like(reverse_mask)
                rect_mask[y_min:y_max + 1, x_min:x_max + 1] = 255

                cropped = img_rare * rect_mask
                cropped = cropped[y_min:y_max + 1, x_min:x_max + 1, :]
            else:
                cropped = img_rare * torch.zeros_like(reverse_mask)

            # 转换为CHW格式并resize
            cropped = cropped.permute(2, 0, 1)
            if mask_indices is not None:
                cropped = self._padding_resize(cropped)

            cropped = F.resize(cropped, (64, 64))
            img_masked.append(cropped / 255.0)

        img_masked = torch.stack(img_masked, dim=0)
        img_masked = img_masked.permute(1, 0, 2, 3)

        return img_masked,img_rare

    def _padding_resize(self, mask):
        """将输入mask填充为正方形

        Args:
            mask: 输入tensor,形状为(C,H,W)

        Returns:
            torch.Tensor: 填充后的正方形tensor
        """
        height, width = mask.shape[1], mask.shape[2]
        max_size = max(height, width)

        if height < max_size:
            # 上下填充
            padding = (0, 0, (max_size - height) // 2, (max_size - height) // 2)
        elif width < max_size:
            # 左右填充
            padding = ((max_size - width) // 2, (max_size - width) // 2, 0, 0)
        else:
            return mask

        padded_mask = F2.pad(mask, padding, value=255)
        return padded_mask
#绘制整张图片


def save_cosine_similarity_heatmap(reg_matrix,folder_path,epoch,name):
    # reg_matrix:matrix
    save_dir = os.path.join(folder_path, f"{epoch}_{name}.png")
    # 设置图像大小和DPI以提高清晰度
    plt.figure(figsize=(10, 8), dpi=100)
    # 2. 将余弦相似度矩阵转换为 numpy 数组
    cos_sim_matrix_np = reg_matrix.clone().detach().cpu().numpy()
    # 在每个单元格中添加文本标注
    for i in range(cos_sim_matrix_np.shape[0]):
        for j in range(cos_sim_matrix_np.shape[1]):
            if cos_sim_matrix_np[i, j] > 0:
                # 在矩阵中添加对应的像素值
                plt.text(j, i, f'{cos_sim_matrix_np[i, j]:.2f}',
                         ha='center', va='center', color='black',
                         fontsize=10)
    # 使用 imshow 绘制热力图
    plt.imshow(cos_sim_matrix_np, cmap='autumn', interpolation='nearest')
    # 显示颜色对应条
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir)
    plt.clf()
    plt.close()






if __name__ == '__main__':
    # `npz` file path is `data/sketch/[DATASET NAME].npz`
    path = './cat.npz'
    # Load the numpy file
    dataset = np.load(str(path), encoding='latin1', allow_pickle=True)
    # print(dataset['train'].shape)
    max_seq_length = 200
    # Create training dataset
    train_dataset = StrokesDataset(dataset['train'], max_seq_length)
    # 使用采样器生成数据
    data, *_ = train_dataset[4:10]
    data = data.transpose(0, 1)



    for i in range(data.shape[1]):
        news=data[:,i,:].clone()
        # # print(news)
        news[:, 0:2]= torch.cumsum(news[:, 0:2], dim=0)

        news[:, 2] = news[:, 3]
        news = news[:, 0:3].detach().cpu().numpy()
        strokes = np.split(news, np.where(news[:, 2] > 0)[0] + 1)

        for s in strokes:

            plt.plot(s[:, 0], -s[:, 1])

        # Show the plot
        plt.savefig(f"sketch_rare_p{i}.png")
        plt.clf()

        datas = rare_process(data[:,i,:].clone())
        strokes = torch.cat(datas, dim=0)
        print(strokes.shape)
        strokes=strokes[:,:2]
        # print(datas)
        img = diffvgProcess(strokes, 32, 32,1)
        img_rare = img.detach().cpu().numpy()
        img_rare = (img_rare * 255).astype(np.uint8)
        img_rare = Image.fromarray(img_rare)

        img_rare.save(f"sketch_rare_d{i}.png")


