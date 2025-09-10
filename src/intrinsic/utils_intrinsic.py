import matplotlib
import torch
import numpy as np
from src.utils.gaussian_model_utils import eval_sh
import torch.nn as nn
import torch.nn.functional as F


def intrinsic_composition(reflectance, shading, offset, residual, view_dirs):#c=(reflect+delta）*shading+residual
    # intrinsic_rgb = reflectance * shading
    intrinsic_rgb = torch.clamp(reflectance + offset, 0, 1) * shading#(reflect+delta）*shading, torch.clamp限制幅度函数，
    
    deg = int(np.sqrt(residual.shape[1]) - 1)#求残差的球谐度数deg
    residual_shs_view = residual.transpose(1, 2).view(-1, 3, residual.shape[1]) # [N, 3, 16] .view()相当于resize，-1表示这里的维度长度取决于其他维度数
    residual_view = eval_sh(deg, residual_shs_view, view_dirs) # [N, 3]
    residual_view = torch.clamp_min(residual_view + 0.5, 0.0)
    intrinsic_rgb = intrinsic_rgb + residual_view#c=(reflect+delta）*shading+residual
    
    return intrinsic_rgb


def rgb2xyz_np(rgb):
    # 将RGB值转换为XYZ空间
    # rgb < 1
    # rgb [N, 3]
    if rgb.max() > 10:
        rgb = rgb / 255.
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb@m.T
    return xyz

def xyz2lab_np(xyz, normalize=True):
    # 将XYZ值转换为Lab空间
    xyz_ref = np.array([0.950456, 1.0, 1.088754])
    
    xyz_ratio = xyz / xyz_ref
    xyz_ratio = np.where(xyz_ratio > 0.008856, xyz_ratio ** (1/3), (903.3 * xyz_ratio + 16) / 116)
    
    lab = np.zeros_like(xyz)
    lab[..., 0] = np.clip(116 * xyz_ratio[..., 1] - 16, 0, 100)
    lab[..., 1] = (xyz_ratio[..., 0] - xyz_ratio[..., 1]) * 500
    lab[..., 2] = (xyz_ratio[..., 1] - xyz_ratio[..., 2]) * 200
    if normalize:
        lab[..., 0] /= 100
        lab[..., 1] = (lab[..., 1] + 128) / 255
        lab[..., 2] = (lab[..., 2] + 128) / 255
    return lab

def rgb2lab_np(rgb):
    xyz = rgb2xyz_np(rgb)
    lab = xyz2lab_np(xyz)
    return lab

def lab2flab(lab, scale=0.3):
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    return torch.stack([l*scale, a, b], dim=-1)

def rgb2xyz(rgb):
    # 将RGB值转换为XYZ空间
    # rgb < 1
    # rgb [N, 3]
    # if rgb.max() > 10:
    #     rgb = rgb / 255.
    if rgb.max() > 10:
        rgb = rgb / 255.
    rgb = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=rgb.device)
    xyz = rgb@m.T
    return xyz

def xyz2lab(xyz, normalize=True):
    xyz_ref = torch.tensor([0.950456, 1.0, 1.088754], device=xyz.device)
    
    xyz_ratio = xyz / xyz_ref
    xyz_ratio = torch.where(xyz_ratio > 0.008856, xyz_ratio ** (1/3), (903.3 * xyz_ratio + 16) / 116)
    
    l = torch.clamp(116 * xyz_ratio[..., 1] - 16, 0,  100)
    a = (xyz_ratio[..., 0] - xyz_ratio[..., 1]) * 500
    b = (xyz_ratio[..., 1] - xyz_ratio[..., 2]) * 200
    
    if normalize:
        l = l / 100
        a = (a + 128) / 255
        b = (b + 128) / 255

    lab = torch.stack([l, a, b], dim=-1)
    return lab

def rgb2lab(rgb):
    xyz = rgb2xyz(rgb)
    lab = xyz2lab(xyz)
    return lab


def cal_gradient(data, p=1):
    """
    data: [1, C, H, W]，看这里调用这个函数的用法，似乎传入的数据都在C维度做了均值化，似乎形状都为[1,1,H,W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    
    # gradient = torch.abs(grad_x) + torch.abs(grad_y)
    grad = torch.cat([grad_x, grad_y], dim=-3)#[1,2,H,W]
    gradient = torch.norm(grad, p=p, dim=-3, keepdim=True)
    return gradient

# def get_sparsity_weight(gt_image, gt_normal, lab, rgb_grad_scale=0.3, normal_grad_scale=0.3, a=80, b=16):
def get_sparsity_weight(gt_image, gt_normal, lab, rgb_grad_scale=0.3, normal_grad_scale=0.3, a=60, b=8.5):
# def get_sparsity_weight(gt_image, gt_normal, lab, rgb_grad_scale=0.3, normal_grad_scale=0.3, a=80, b=10):
    lab_feature = torch.stack([lab[0]*0.3, lab[1], lab[2]])
    # lab_norm = torch.norm(lab_feature,p=1, dim=0, keepdim=True)
    # lab_grad = cal_gradient((lab_norm**1).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    lab_grad = cal_gradient((lab_feature).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    rgb_grad = cal_gradient(gt_image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    lab_rgb_grad = torch.where(lab_grad > rgb_grad_scale * rgb_grad , lab_grad, rgb_grad_scale * rgb_grad)#torch.where(condition, x, y)根据condition选择x或者y
    
    normal = (gt_normal + 1) / 2 # [-1, 1] -> [0, 1]
    normal_grad = cal_gradient(normal.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    lab_rgb_normal_grad = torch.where(lab_rgb_grad > normal_grad_scale * normal_grad, lab_rgb_grad, normal_grad_scale * normal_grad)
    
    # 1 / (1 + exp(ax-b))
    lab_rgb_weight = 1 / (1 + torch.exp(a * lab_rgb_normal_grad - b))
    
    return lab_rgb_weight

def get_smooth_weight(gt_depth, gt_normal, gt_lab, depth_threshold=0.1, normal_threshold=0.2):
    lab_feature = torch.stack([gt_lab[0]*0.3, gt_lab[1], gt_lab[2]])
    # lab_norm = torch.norm(lab_feature,p=2, dim=0, keepdim=True)
    # lab_grad = cal_gradient((lab_norm**2).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    lab_grad = cal_gradient((lab_feature).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)

    #TODO 为什么直接用lab_grad作为权重，而不是下面这个weight？ 答：这个depth_grad没用，本文的normal_grad在调用get_smooth_weight之后再计算的，实现这个代码的论文似乎有depth_grad，但是本文不用depth就有此实现上的区别
    # depth_grad = cal_gradient(gt_depth.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    # normal_grad = cal_gradient(gt_normal.abs().mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    # smooth_mask = torch.where(normal_grad < normal_threshold, True, False)
    # weight = normal_grad[smooth_mask]
    
    return lab_grad

def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)

def check_nan_in_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")
        print(tensor)
        return True
    return False

