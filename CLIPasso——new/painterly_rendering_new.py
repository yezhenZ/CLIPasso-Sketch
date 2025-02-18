import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import time
import traceback

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
import torchvision

import config
import sketch_utils as utils
from models.loss import Loss
from bezier_renderer import BezierRenderer
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
import model_rnngcn
from torch.utils.tensorboard import SummaryWriter
from data_utils import compute_cosine_similarity
def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # 利用 U2Net 模型生成图像的蒙版 mask，遮蔽后的图像 masked_im
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)
    # 图像缩放调整
    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask

#计算余弦相似度


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter grad:\n{param.grad}")  # 打印参数的值
        print(f"Gradient:\n{param.requires_grad}")  # 打印参数的梯度
        print("-" * 50)  # 分隔线，用来分隔每个参数


def init_writer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, "runs")
    save_path = os.path.join(base_dir, "cos_matrix")
    img_dir = os.path.join(base_dir, "render_img")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path in [runs_dir, save_path]:
        os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(runs_dir)
    return writer, runs_dir, save_path,img_dir



def main(args):
    mseloss = torch.nn.MSELoss()
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)

    writer, runs_dir, save_path ,img_dir= init_writer()
    #创建模型
    cnnModel = model_rnngcn.SimpleCNN().to(args.device)
    GCNmodel = model_rnngcn.GCN(input_dim=128, output_dim=4).to(args.device)  # 4 是每条bezier曲线控制点的数量
    optimizer = PainterOptimizer(args, list(cnnModel.parameters()),list(GCNmodel.parameters()), renderer)

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

    # not using tdqm for jupyter demo
    if args.display:

        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))



    for epoch in epoch_range:

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
        points_rare=torch.stack(renderer.control_points_set,dim=0)
        points_rare=points_rare.clone().detach()
        # print(points_rare.shape)
        # 生成掩码图像
        bezier_renderer = BezierRenderer(224, 224)
        bezier_masked,img_rare = bezier_renderer.mask_img(renderer.control_points_set)

        #打印mask
        img_masked = bezier_masked.permute(1, 0, 2, 3)
        img_masked_grad = torchvision.utils.make_grid(img_masked, nrow=8, padding=2)

        # 特征提取和GCN处理,new_points_test是8
        feature,new_points_test = cnnModel(bezier_masked)
        feature=feature.view(-1,128)
        reg_matrix, cos_matrix = compute_cosine_similarity(feature)

        new_points_test = torch.sigmoid(new_points_test).view(16, -1, 2)
        new_points_test = new_points_test*224
        # print(feature.shape)
        new_points = GCNmodel(feature, reg_matrix).view(-1, 4, 2)
        # print(new_points)
        sketches = utils.render_img_rgb_from_renderer(new_points, renderer,epoch,img_dir).to(args.device)
        ##


        Mseloss=mseloss(new_points_test,points_rare)
        # writer.add_image(f"{epoch}_sketch", sketches_new)
        losses_dict = loss_func(sketches, inputs.detach(), renderer.get_color_parameters(), renderer, counter, optimizer)
        if epoch <= 100:
            loss=Mseloss
        else:
            cliploss=sum(list(losses_dict.values()))
            loss = cliploss+Mseloss*0.01
        loss.backward()
        optimizer.step_()
        max_grad_norm=1.0
        torch.nn.utils.clip_grad_norm_(cnnModel.parameters(),max_grad_norm)
        if epoch % args.save_interval == 0:
            # 打印网格的形状,chw
            utils.save_cosine_similarity_heatmap(cos_matrix, save_path, epoch, "cos_matrix")
            writer.add_image(f'{epoch}images_grid', img_masked_grad)
            writer.add_image(f'{epoch}img_rare', img_rare.permute(2,0,1))

            # 保存为 .pt 文件（PyTorch 张量格式）
            torch.save(feature, f"{args.output_dir}/feature.pt")
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
        if epoch % args.eval_interval == 0:

            # for i ,path in enumerate(renderer.shapes):
            #     print(f"the {i}grad are {path.points.grad}")
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
                if epoch <= 100:
                    print(
                        f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")
                    # 保存loss情况
                    writer.add_scalar("loss_total.", loss.item(), global_step=epoch)
                else:
                    print(
                        f"eval iter[{epoch}/{args.num_iter}] loss_total[{loss.item()}] time[{time.time() - start}],"
                        f"mseloss[{Mseloss.item()}] ,"
                        f"cliploss[{cliploss.item()}]"
                    )
                    # 保存loss情况
                    writer.add_scalar("loss_total.", loss.item(), global_step=epoch)
                    writer.add_scalar("mseloss.", Mseloss.item(), global_step=epoch)
                    writer.add_scalar("cliploss.", cliploss.item(), global_step=epoch)



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

            # if abs(cur_delta) <= min_delta:
            #     if terminate:
            #         break
            #     terminate = True

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

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save
    # 计算s‘



        # img_grid = torchvision.utils.make_grid(img_masked, nrow=8, padding=2)
        # img_beziers_grid = torchvision.utils.make_grid(img_beziers_clone, nrow=8, padding=2)


            # print(feature)
        # print(reg_matrix.shape)
        # cos_matrix=reg_matrix.clone()

        # new_points = GCNmodel(new_points, reg_matrix).view(-1, 4, 2)
        # print(feature)
        # sketches=display_img.render_sketch(new_points,renderer,epoch).to(args.device)

        # sketches_new=sketches.clone().squeeze(0)
        # writer.add_image(f"{epoch}_sketch", sketches_new)

        # if epoch%10==0:
        #     for i, path in enumerate(renderer.shapes):
        #         print(f"the {i}grad are {path.points.grad}")
        # # 打印 cnnModel 和 GCNmodel 的参数
        # print("cnnModel Parameters:")
        # print_model_parameters(cnnModel)
        # print(" Parameters:")

#任务：改变svg绘制，修改并加入mask，定时画出余弦相似度矩阵热力图



if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()


