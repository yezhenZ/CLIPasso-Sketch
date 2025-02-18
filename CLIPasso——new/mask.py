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
import bezier_renderer
import sketch_utils as utils
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2
#绘制图片,一条bezier曲线，4个控制点，三阶
def render(control_points,canvas_width,canvas_height,stroke_width):
    list_control_points=control_points #tensor
    num_points=2
    # list_control_points=control_points.points.contiguous()
    path = pydiffvg.Path(
        num_control_points=torch.tensor([num_points]),
        points=list_control_points,
        stroke_width=torch.tensor(stroke_width),
        is_closed=False
    )
    shapes=[path]
    # Create shape group
    shape_group= pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=None
        , stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32), #黑色
        #  fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0], dtype=torch.float32)
    )
    shape_groups=[shape_group]
    # Serialize the scene
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    # # 背景图像
    background_image = torch.ones(canvas_height, canvas_width, 4)
    background_image[:, :, 0:3] = 1.0  # 设置RGB通道为1，表示白色
    background_image[:, :, 3] = 1.0  # 设置A通道为1，表示完全不透明background_image,

    # Render the image
    render = pydiffvg.RenderFunction.apply

    img = render(canvas_width, canvas_height, 2, 2, 0, background_image, *scene_args)

    gray_img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray_img = gray_img.unsqueeze(2)

    return gray_img

if __name__ == '__main__':
    rare_points = torch.load(
        f"/home/dhu/zyl/CLIPasso/CLIPasso-main/output_sketches_rare/camel/camel_16strokes_seed2000/control_points_epoch.pt")
    test_rare = torch.stack(rare_points, dim=0)
    test = test_rare[0]
    #1.正方形区域mask
    #绘制原始图片
    img_test=display_img.diffvgProcess(test_rare,224,224)
    #绘制单个curve，curve粗细要保持一致
    img_mask=render(test,224,224,2)
    # test_list=display_img.mask_img(img_test,img_mask)

    # #二值化mask，黑色0,255白色
    binary_mask = (img_mask>0).float()  # 将值大于0的部分设置为1，小于等于0的部分设置为0
    binary_mask = binary_mask * 255  # 将 1 的部分设为 255，0 的部分设为 0    #
    # 反转 mask，用于找到黑色（值为0）的边界
    reverse_mask = 255 - binary_mask  # 反转掩膜，黑色变为白色，白色变为黑色,黑色是0
    # 计算非零域的边界（我们可以用 torch 的 max 和 min 函数来模拟找到外接矩形的过程）
    test=reverse_mask
    # for i in range(test.shape[0]):
    #     for j in range(test.shape[1]):
    #         if test[i,j]!=0:
    #             print(i,j,test[i,j])
    mask_nonzero = reverse_mask.view(-1)  # 将 mask 展平成一维
    indices = mask_nonzero.nonzero().squeeze() # 获取非零元素的索引

    if len(indices) > 0:
        rows = indices // binary_mask.shape[1]  # 每个非零元素的行索引
        cols = indices % binary_mask.shape[1]  # 每个非零元素的列索引

        y_min = rows.min()
        y_max = rows.max()
        x_min = cols.min()
        x_max = cols.max()
        # 生成 mask，只保留外接矩形部分
        mask = torch.zeros_like(reverse_mask)
        mask[y_min:y_max, x_min:x_max] = 255
        image = img_test * mask
        # 使用切片操作在 image 上截取区域
        cropped_image = image[ y_min:y_max, x_min:x_max,:]


    else:
        mask = torch.zeros_like(reverse_mask)
        cropped_image = img_test * mask

    cropped_image = cropped_image.permute(2, 0, 1)
    # 转换大小为 (224, 224)
    cropped_image_resized = F.resize(cropped_image, (64, 64))

    cropped_image_gray = cropped_image_resized / 255.0  # 如果原图是 0-255 的像素值
    #

    # transform = transforms.Compose([transforms.Resize(size=(224, 224)),
    #                                 transforms.Grayscale(num_output_channels=1)]) # 确保输出为单通道])
    # cropped_image=transform(cropped_image)
    cropped_image = cropped_image_gray.permute(1, 2, 0)

    image = cropped_image.cpu().detach().numpy()
    plt.imshow(image, cmap='gray')  # 使用灰度色图显示
    # plt.axis('off')  # 关闭坐标轴显示
    plt.show()
    plt.savefig("test0.png")
    # plt.imshow(img_test, cmap='gray')  # 使用灰度色图显示
    # # plt.axis('off')  # 关闭坐标轴显示
    # plt.show()
    # plt.savefig("test1.png")
    # # 3. 使用掩码提取指定区域
    # masked_img = img_mask.cpu().detach() * mask  # 只保留小于 0.999 的部分（值为 1 的地方保留，其他为 0）
    # print(f"Masked image shape: {mask.shape}")
    # 将张量转换为 NumPy 数组

