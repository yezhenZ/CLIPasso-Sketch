import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import wandb
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm, trange
import torchvision
from torch.utils.tensorboard import SummaryWriter

import config
import sketch_utils as utils
from models.self_model import SimpleCNN
from models.self_model import GCN
from bezier_renderer import BezierRenderer
from models.loss import Loss
from models.painter_params_modified import Painter, PainterOptimizer
import matplotlib.pyplot as plt

from data_utils import compute_cosine_similarity,check_gradients

def render_img_from_renderer(points, renderer, epoch, save_path):
    img = utils.render_img_rgb_from_renderer(points, renderer)
    
    if epoch % 10 == 0:
        save_dir = os.path.join(save_path, f"{epoch}_out.png")
        # print(img2.shape)
        img2 = img.clone().detach().cpu().numpy()
        # 使用 matplotlib 绘制图像
        plt.imshow(img2)
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_dir)
        plt.clf()
        plt.close()
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to(renderer.device)  # NHWC -> NCHW
    return img

def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                      num_segments=args.num_segments,
                      imsize=args.image_scale,
                      device=args.device,
                      target_im=target_im,
                      mask=mask)
    return renderer.to(args.device)

def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        new_image = Image.new("RGBA", target.size, "WHITE")
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    
    # 生成蒙版和遮蔽图像
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)
        
    # 图像预处理
    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize((args.image_scale, args.image_scale), 
                                          interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(args.image_scale, 
                                          interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    
    target_ = transforms.Compose(transforms_)(target).unsqueeze(0).to(args.device)
    return target_, mask



def train(epoch, args, renderer: Painter, optimizer: PainterOptimizer, loss_func, inputs, 
                   cnn_model, gcn_model, writer, save_path, counter,epoch_range):
    """训练一个epoch"""
    if not args.display:
        epoch_range.refresh()
    renderer.set_random_noise(epoch)
    if args.lr_scheduler:
        optimizer.update_lr(counter)
        
    start = time.time()
    optimizer.zero_grad_()

    # 获取控制点
    for i, path in enumerate(renderer.shapes):
        renderer.control_points_set[i] = path.points
    
    points_orig = torch.stack(renderer.control_points_set, dim=0)
    points_orig = points_orig.clone().detach()
    
    # 生成掩码图像
    bezier_renderer = BezierRenderer(224,224)
    bezier_masked, img_orig = bezier_renderer.mask_img(renderer.control_points_set)
    
    img_masked = bezier_masked.permute(1,0,2,3)
    img_masked_grid = torchvision.utils.make_grid(img_masked, nrow=8, padding=2)
    
    # 特征提取和GCN处理
    feature, control_points_set_hat = cnn_model(bezier_masked)
    feature = feature.view(-1, 128)
    reg_matrix, cos_matrix = compute_cosine_similarity(feature)
    
    control_points_set_hat = torch.sigmoid(control_points_set_hat).view(16,-1,2)
    control_points_set_hat = control_points_set_hat * renderer.canvas_width
    
    new_points = gcn_model(feature, reg_matrix).view(-1, 4, 2)
    
    sketches = render_img_from_renderer(new_points, renderer, epoch, save_path).to(args.device)
    
    mes_loss = torch.nn.MSELoss()(control_points_set_hat, points_orig)
    # 计算损失并反向传播
    losses_dict = loss_func(sketches, inputs.detach(),
                          renderer.get_color_parameters(), renderer, counter, optimizer)
    if epoch <= 100:
        loss = mes_loss
    else:
        loss = sum(list(losses_dict.values())) + mes_loss
    loss.backward()
    
    optimizer.step_()
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_grad_norm)
    
    # 保存中间结果
    if epoch % args.save_interval == 0:
        utils.save_cosine_similarity_heatmap(cos_matrix, save_path, epoch, "cos_matrix")
        
        writer.add_image(f'{epoch}images_grid', img_masked_grid,dataformats='HWC')
        writer.add_image(f'{epoch}img_orig', img_orig.permute(2,0,1))
        
        torch.save(feature, f"{args.output_dir}/feature.pt")

        utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                        use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
        renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
        
    return loss, losses_dict, sketches


def init_writer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, "runs")
    save_path = os.path.join(base_dir, "cos_matrix")
    
    for path in [runs_dir, save_path]:
        os.makedirs(path, exist_ok=True)
        
    writer = SummaryWriter(runs_dir)
    return writer, runs_dir, save_path

def main(args):
    # 初始化
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)
    
    writer, runs_dir, save_path = init_writer()
    
    # 创建模型
    cnn_model = SimpleCNN().to(args.device)
    gcn_model = GCN(input_dim=128, output_dim=4).to(args.device)
    model_parameters = list(cnn_model.parameters()) + list(gcn_model.parameters())
    optimizer = PainterOptimizer(args, renderer, model_parameters)
    
    # 初始化训练
    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    
    min_delta = 1e-5
    terminate = False
    
    epoch_range = range(args.num_iter) if args.display else tqdm(range(args.num_iter))
    
    # 训练循环
    for epoch in epoch_range:
        loss, losses_dict, sketches = train(epoch, args, renderer, optimizer, loss_func,
                                          inputs, cnn_model, gcn_model, writer, 
                                          save_path, counter, epoch_range)
        
        # 评估和记录
        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    # terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)
                    
        counter += 1
        
    # 保存最终结果
    renderer.save_svg(args.output_dir, "final_svg")
    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    
    try:
        configs_to_save = main(args)
        for k in configs_to_save.keys():
            final_config[k] = configs_to_save[k]
        np.save(f"{args.output_dir}/config.npy", final_config)
        
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
        
    if args.use_wandb:
        wandb.finish()
