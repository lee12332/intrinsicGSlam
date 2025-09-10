#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from src.utils.gaussian_model_utils import (RGB2SH, build_scaling_rotation,
                                            inverse_sigmoid,strip_symmetric)


class GaussianModel:
    def __init__(self, sh_degree: int = 3, use_intrinsic=True,isotropic=False):
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "scaling",
            "rotation",
            "opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0, 4).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        # self.percent_dense = 0
        self.spatial_lr_scale = 1

        self.isotropic = isotropic
        self.use_intrinsic = use_intrinsic
        self.setup_functions()

        if self.use_intrinsic:
            self._reflectance = torch.empty(0).cuda()
            self._intensity = torch.empty(0).cuda() #shading
            self._residual_dc = torch.empty(0).cuda() #这个应该就是与方向相关的颜色，[n,3,16],16是球谐系数，3是什么？
            self._residual_rest = torch.empty(0).cuda()
            self._light = torch.empty(0).cuda()
            self._offset = torch.empty(0).cuda()
        
            self.gaussian_param_names += ['reflectance', 'intensity','residual_dc', 'residual_rest', 'light', 'offset']
 

    def restore_from_params(self, params_dict, training_args):
        self.training_setup(training_args)
        self.densification_postfix(
            params_dict["xyz"],
            params_dict["features_dc"],
            params_dict["features_rest"],
            params_dict["opacity"],
            params_dict["scaling"],
            params_dict["rotation"])

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if self.use_intrinsic:
            self.reflectance_activation = torch.sigmoid
            self.intensity_activation = torch.sigmoid
            self.light_activation = torch.relu

    def capture_dict(self):
        captDict = {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.clone().detach().cpu(),
            "features_dc": self._features_dc.clone().detach().cpu(),
            "features_rest": self._features_rest.clone().detach().cpu(),
            "scaling": self._scaling.clone().detach().cpu(),
            "rotation": self._rotation.clone().detach().cpu(),
            "opacity": self._opacity.clone().detach().cpu(),
            "max_radii2D": self.max_radii2D.clone().detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().cpu(),
            "denom": self.denom.clone().detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_intrinsic:
            intrinsic_dict = {
                "reflectance" :self._reflectance.clone().detach().cpu(),
                "intensity" : self._intensity.clone().detach().cpu(),
                "residual_dc" : self._residual_dc.clone().detach().cpu(),
                "residual_rest" : self._residual_rest.clone().detach().cpu(),
                "light" : self._light.clone().detach().cpu(),
                "offset" : self._offset.clone().detach().cpu(),
            }
            captDict.update(intrinsic_dict)
        return captDict

    def get_size(self):
        return self._xyz.shape[0]

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[:, 0:1]  # Extract the first column
            scales = scale.repeat(1, 3)  # Replicate this column three times
            return scales
        return self.scaling_activation(self._scaling)

    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_active_sh_degree(self):
        return self.active_sh_degree

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(self.get_scaling(), scaling_modifier, self._rotation)

    def get_reflectance(self):
        return self.reflectance_activation(self._reflectance)
    
    def get_light(self):
        return self._light
    
    def get_shading(self):
        return self.intensity_activation(self._intensity)
    
    def get_residual(self):
        residual_dc = self._residual_dc
        residual_rest = self._residual_rest
        return torch.cat((residual_dc, residual_rest), dim=1)
    
    def get_offset(self):
        return self._offset
    
    def add_points(self, pcd: o3d.geometry.PointCloud): #初始化了新增点云的属性
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        new_points_num = fused_point_cloud.shape[0]
        fused_color = RGB2SH(torch.tensor(
            np.asarray(pcd.colors)).float().cuda())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()) # [P, 3, (sh_degree + 1) ** 2] 
        features[:, :3, 0] = fused_color
        print("Number of added points: ", new_points_num)

        global_points = torch.cat((self.get_xyz(),torch.from_numpy(np.asarray(pcd.points)).float().cuda()))
        dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001) #distCUDA2计算点云中的每个点到与其最近的k个点距该点的平均平方距离，这里k=3
        dist2 = dist2[self.get_size():] #新增点对所有点的distMin

        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 3) #论文3.2节，scale是根据活动子地图内的最近邻距离定义的；[..., None]增加一个维度[n,1];repeat(1, 3)在第一个维度复制为1倍,在第二个维度复制为3倍；[n,3]
        # scales = torch.log(0.001 * torch.ones_like(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((new_points_num, 4), device="cuda")
        rots[:, 0] = 1 #[n,4]，值[1,0,0,0]
        opacities = inverse_sigmoid(0.5 * torch.ones((new_points_num, 1), dtype=torch.float, device="cuda"))#[n,1];值[inverse_sigmoid（0.5）]

        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))

        if self.use_intrinsic:
            reflectance = torch.zeros_like(fused_point_cloud)
            intensity = torch.ones((new_points_num, 1), dtype=torch.float, device="cuda")
            residual = torch.zeros((new_points_num, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            light = torch.ones_like(fused_point_cloud)
            offset = torch.zeros_like(fused_point_cloud)
                                         
            new_reflectance = nn.Parameter(reflectance.requires_grad_(True))
            new_intensity = nn.Parameter(intensity.requires_grad_(True))
            new_residual_dc = nn.Parameter(residual[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            new_residual_rest = nn.Parameter(residual[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            new_light = nn.Parameter(light.requires_grad_(True))
            new_offset = nn.Parameter(offset.requires_grad_(True))

        d={
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.use_intrinsic:
            d_intrinsic = { "reflectance": new_reflectance,
                "intensity": new_intensity,
                "residual_dc": new_residual_dc,
                "residual_rest": new_residual_rest,
                "light": new_light,
                "offset": new_offset,}
            d.update(d_intrinsic)

        self.densification_postfix(d)

    def training_setup(self, training_args):
        # self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if self.use_intrinsic:
            params += [
                {'params': [self._reflectance], 'lr': training_args.reflectance_lr, "name": "reflectance"},
                {'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"},
                {'params': [self._residual_dc], 'lr': training_args.residual_lr, "name": "residual_dc"},
                {'params': [self._residual_rest], 'lr': training_args.residual_rest_lr, "name": "residual_rest"},
                {'params': [self._light], 'lr': training_args.light_lr, "name": "light"},
                {'params': [self._offset], 'lr': training_args.offset_lr, "name": "offset"},]

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def training_setup_camera(self, cam_rot, cam_trans, cfg):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        params = [ #优化的参数，是一个多字典组成的列表
            {"params": [self._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [self._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [self._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [self._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0.0, "name": "rotation"},
            {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
                "name": "cam_unnorm_rot"},
            {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
                "name": "cam_trans"},
        ]

        if self.use_intrinsic:
            params += [
                {'params': [self._reflectance], 'lr': 0.0, "name": "reflectance"},
                {'params': [self._intensity], 'lr': 0.0, "name": "intensity"},
                {'params': [self._residual_dc], 'lr': 0.0, "name": "residual_dc"},
                {'params': [self._residual_rest], 'lr': 0.0, "name": "residual_rest"},
                {'params': [self._light], 'lr': 0.0, "name": "light"},
                {'params': [self._offset], 'lr': 0.0, "name": "offset"},]
            
        self.optimizer = torch.optim.Adam(params, amsgrad=True)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(self._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 3))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
                axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_intrinsic:
            self._reflectance = optimizable_tensors["reflectance"]
            self._intensity = optimizable_tensors["intensity"]
            self._residual_dc = optimizable_tensors["residual_dc"]
            self._residual_rest = optimizable_tensors["residual_rest"]
            self._light = optimizable_tensors["light"]
            self._offset = optimizable_tensors["offset"]


    def cat_tensors_to_optimizer(self, tensors_dict): #将新gs点加入优化器中
        """
        优化器参数的格式参考
        params = [
            {"params": [self._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [self._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [self._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [self._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0.0, "name": "rotation"},
            # {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
            #     "name": "cam_unnorm_rot"},
            # {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
            #     "name": "cam_trans"},
        ]
        if self.use_intrinsic:
            params += [
                {'params': [self._reflectance], 'lr': 0.0, "name": "reflectance"},
                {'params': [self._intensity], 'lr': 0.0, "name": "intensity"},
                {'params': [self._residual_dc], 'lr': 0.0, "name": "residual_dc"},
                {'params': [self._residual_rest], 'lr': 0.0, "name": "residual_rest"},
                {'params': [self._light], 'lr': 0.0, "name": "light"},
                {'params': [self._offset], 'lr': 0.0, "name": "offset"},]
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: #遍历每个参数的字典
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)# self.optimizer.state存储了优化器为每个参数维护的状态字典，键是参数张量本身，值是包含该参数优化状态的字典；get（）从state字典中安全地获取键对应的值，没有就是none
            if stored_state is not None: #该参数已经参与过优化（不是子图的第一帧）
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group["params"][0]] #删除已经优化过的参数在优化器状态字典中的记录
                group["params"][0] = nn.Parameter( #更新参数为已经优化的参数cat新添加的参数
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state #更新新参数的状态字典

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def densification_postfix(self, d:dict):
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros(
            (self.get_xyz().shape[0]), device="cuda")
        
        if self.use_intrinsic:
            self._reflectance = optimizable_tensors["reflectance"]
            self._intensity = optimizable_tensors["intensity"]
            self._residual_dc = optimizable_tensors["residual_dc"]
            self._residual_rest = optimizable_tensors["residual_rest"]
            self._light = optimizable_tensors["light"]
            self._offset = optimizable_tensors["offset"]

