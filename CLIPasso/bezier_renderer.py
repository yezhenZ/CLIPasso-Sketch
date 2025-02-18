import pydiffvg
from torchvision.transforms import functional as F
import torch.nn.functional as F2
import torch

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
        """将贝塞尔曲线控制点转换为渲染图像
        
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

    def mask_img(self, control_points):
        """生成每条曲线的mask图像
        
        Args:
            control_points: 所有曲线的控制点
            
        Returns:
            torch.Tensor: 每条曲线对应的mask图像,CHW格式,大小64x64
        """
        img_orig = self.render_img_raw(control_points)
        img_beziers = self.render_beziers(control_points)
        
        img_masked = []
        for img_mask in img_beziers:
            # 生成二值mask,注意不要乘以255导致梯度消失
            binary_mask = (img_mask > 0).float()
            reverse_mask = 1 - binary_mask
            
            # 获取非零区域的边界框
            mask_indices = self._get_nonzero_indices(reverse_mask)
            if mask_indices is not None:
                y_min, y_max, x_min, x_max = mask_indices
                
                # 生成矩形mask并裁剪图像
                rect_mask = torch.zeros_like(reverse_mask)
                rect_mask[y_min:y_max+1, x_min:x_max+1] = 1.0
                
                cropped = img_orig * rect_mask
                cropped = cropped[y_min:y_max+1, x_min:x_max+1, :]
            else:
                cropped = img_orig * torch.zeros_like(reverse_mask)
                
            # 转换为CHW格式并resize
            cropped = cropped.permute(2, 0, 1)
            if mask_indices is not None:
                cropped = self._padding_resize(cropped)
                
            cropped = F.resize(cropped, (64, 64))
            img_masked.append(cropped)  # 不除以255保持梯度
            
        img_masked = torch.stack(img_masked, dim=0)
        img_masked = img_masked.permute(1, 0, 2, 3)

        return img_masked, img_orig

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






