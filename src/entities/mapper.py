""" This module includes the Mapper class, which is responsible scene mapping: Paragraph 3.2  """
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from torchvision.utils import save_image, make_grid
import os

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *  # noqa - needed for debugging

from src.utils.utils import get_render_settings_intrinsic,render_gaussian_model_intrinsic

from src.intrinsic.utils_intrinsic import (cal_gradient,get_sparsity_weight,
                                           get_smooth_weight,lab2flab,rgb2lab,
                                           visualize_depth,check_nan_in_tensor)

from kornia.geometry import depth_to_normals
                                        



class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]#非新子图第一帧的迭代次数
        self.new_submap_iterations = config["new_submap_iterations"]#新子图第一帧的迭代次数
        self.new_submap_points_num = config["new_submap_points_num"] #新子图，对没有点的子图的第一帧中添加gs的数量
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"] #子图中有点，对该帧中添加gs的数量
        self.new_points_radius = config["new_points_radius"] #判断seed的新gs与视锥内的gs的最近邻距离是否小于改阈值，改阈值之内没有最近邻点，就是要添加的点
        self.alpha_thre = config["alpha_thre"] #用于判断不完全重建
        self.pruning_thre = config["pruning_thre"]#用于剔除
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = [] #这是所有的keyframe的列表，结构：keyframes=[(frame_id, keyframe)],其中keyframe={"color": ,"depth": ,"render_settings:"}

    
    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        #基于阿尔法掩模或颜色梯度计算播种新高斯模型的区域
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap: #新子图，第一帧的密实化：使用颜色梯度的边缘mask来添加gs
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:#子图的非第一帧的密实化：在重建不完全区域和几何异常区域来添加gs
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)#alpha值低于阈值的mask区域（重建不完全区域）
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median()) #渲染深度大于真实深度（需要在当前渲染深度前，真实深度处添加gs） 且 深度损失大于40*损失中值（几何误差大）
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask
    

    def compute_seeding_mask_intrinsic(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        seeding_mask = None
        if new_submap: #新子图，第一帧的密实化：使用颜色梯度的边缘mask来添加gs
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:#子图的非第一帧的密实化：在重建不完全区域和几何异常区域来添加gs
            render_dict = render_gaussian_model_intrinsic(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)#alpha值低于阈值的mask区域（重建不完全区域）
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median()) #渲染深度大于真实深度（需要在当前渲染深度前，真实深度处添加gs） 且 深度损失大于40*损失中值（几何误差大）
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)
        """
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w) #这里的depth为什么要乘1.005
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask)  # 获取seeding_mask的非零像素索引
        if is_new_submap: #新子图，添加gs的索引参考了三个：valid_ids（seeding_mask），uniform_ids，gradient_ids
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False) #从pts中随机选择new_submap_points_num个数，不可重复，这里从pts的点数索引中选出了指定数量的索引
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num) #基于图像梯度的模采样像素索引
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            sample_ids = np.unique(combined_ids) # seedmask的索引valid_ids（重建不足，错误率高，颜色边缘的索引），随机索引uniform_ids，颜色梯度索引gradient_ids
        else: #不是新子图，添加索引valid_ids（seeding_mask）
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]] #这个索引操作很慢，后面可以试试改成乘法
        return pts[sample_ids, :].astype(np.float32)

    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}

        current_frame_iters = self.current_view_opt_iterations * iterations #当前帧需要优化的次数
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()
        while iteration < iterations + 1:
            gaussian_model.optimizer.zero_grad(set_to_none=True) #梯度清零
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            # f_mask=torch.sum(~mask).item() #坏点的个数
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = isotropic_loss(gaussian_model.get_scaling()) #各项同性的正则化损失
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():
                if iteration == iterations // 2 or iteration == iterations: #优化到一半或者结束的时候，剔除
                    prune_mask = (gaussian_model.get_opacity()
                                  < self.pruning_thre).squeeze()
                    gaussian_model.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step() #更新参数
                # print(gaussian_model.optimizer.state_dict()['param_groups'][0]['lr'])
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict


    def optimize_submap_intrinsic(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100) -> dict:
        iteration = 0
        losses_dict = {}

        current_frame_iters = self.current_view_opt_iterations * iterations #当前帧需要优化的次数
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()
        while iteration < iterations + 1:
            gaussian_model.optimizer.zero_grad(set_to_none=True) #梯度清零
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model_intrinsic(gaussian_model, keyframe["render_settings"])

            #render_pkg:color,depth,radii,alpha,intrinsic,reflectance,shading,offset,residual
            gt_image = keyframe["color"].cuda()#[3,H,W]
            gt_depth = keyframe["depth"].cuda()#[H,W]
            gt_normal=keyframe["normal"].cuda()#[3,H,W]
            gt_lab=keyframe["lab"].cuda()#[3,H,W]
            # rendered_image= render_pkg["color"]
            rendered_intrinsic=render_pkg["intrinsic"]
            rendered_depth = render_pkg["depth"]
            rendered_reflectance = render_pkg["reflectance"]
            rendered_shading = render_pkg["shading"]
            rendered_offset = render_pkg["offset"]
            rendered_residual = render_pkg["residual"]

            depth_mask = (gt_depth > 0) & (~torch.isnan(rendered_depth)).squeeze(0)
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                rendered_intrinsic[:, depth_mask], gt_image[:, depth_mask]) + self.opt.lambda_dssim * (1.0 - ssim(rendered_intrinsic, gt_image))
            # check_nan_in_tensor(color_loss,"1")
            depth_loss = l1_loss(rendered_depth[:, depth_mask], gt_depth[depth_mask])
            # check_nan_in_tensor(depth_loss,"2")
            reg_loss = isotropic_loss(gaussian_model.get_scaling()) #各项同性正则化损失
            # check_nan_in_tensor(reg_loss,"3")
            total_loss = color_loss + depth_loss + reg_loss
            # total_loss = color_loss + depth_loss 
            # check_nan_in_tensor(total_loss,"1")

            # total_loss = color_loss + reg_loss
            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                    }

            #intrinsic loss
            lambda_sparsity=self.config["lambda_reflectance_sparsity"]
            if lambda_sparsity > 0:#反射率分段稀疏
                sparsity_weight= get_sparsity_weight(gt_image, gt_normal, gt_lab)
                reflectance_grad = cal_gradient(rendered_reflectance.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
                # sparsity_loss = (sparsity_weight * reflectance_grad).mean()
                sparsity_loss = (sparsity_weight * reflectance_grad).mean() #[1,H,W]
                total_loss = total_loss + sparsity_loss * lambda_sparsity
                # check_nan_in_tensor(total_loss,"2")

                losses_dict[frame_id].update({"sparsity_loss":sparsity_loss.item() * lambda_sparsity}) 
            
            lambda_smooth=self.config["lambda_shading_smooth"]
            if lambda_smooth > 0:#照明平滑
                smooth_weight = get_smooth_weight(gt_depth, gt_normal, gt_lab)
                shading_grad = cal_gradient(rendered_shading.mean(0, keepdim=True).unsqueeze(0), p=2).squeeze(0) **2
                normal_grad = cal_gradient(gt_normal.abs().mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
                smooth_mask = torch.where(normal_grad < 0.1, True, False)
                shading_smooth_loss = (smooth_weight * shading_grad)[smooth_mask].mean()#[1,H,W]
                total_loss = total_loss + lambda_smooth * shading_smooth_loss
                # check_nan_in_tensor(total_loss,"3")

                losses_dict[frame_id].update({"shading_smooth_loss":shading_smooth_loss.item() * lambda_smooth})
    
            lambda_chrom=self.config["lambda_prior_chromaticity"]
            if lambda_chrom > 0:#反射率色度损失
                prior_chrom = lab2flab(rgb2lab(gt_image.permute(1, 2, 0)))
                reflect = torch.clamp(rendered_reflectance + rendered_offset, 0, 1)
                reflect_chrom = lab2flab(rgb2lab(reflect.permute(1, 2, 0)))
                loss_prior_chromaticity = ((prior_chrom - reflect_chrom) ** 2).mean() #[H,W,3]
                total_loss = total_loss + loss_prior_chromaticity * lambda_chrom
                # check_nan_in_tensor(total_loss,"4")

                losses_dict[frame_id].update({"prior_chromaticity_loss" : loss_prior_chromaticity.item() * lambda_chrom})

            lambda_residual=self.config["lambda_residual"]
            if lambda_residual > 0:
                loss_residual = (torch.norm(rendered_residual, dim=0)**2).mean()#[3,H,W]
                total_loss = total_loss + lambda_residual * loss_residual
                # check_nan_in_tensor(total_loss,"5")

                losses_dict[frame_id].update({"residual_loss" : loss_residual.item() * lambda_residual})

            lambda_offset=self.config["lambda_offset"]
            if lambda_offset > 0:
                loss_offset = (torch.norm(rendered_offset, dim=0)**2).mean() #[3,H,W]
                total_loss = total_loss + lambda_offset * loss_offset
                # check_nan_in_tensor(total_loss,"6")

                losses_dict[frame_id].update({"offset_loss" : loss_offset.item() * lambda_offset})
            
            losses_dict[frame_id].update({"total_loss":total_loss.item()}) 
            # check_nan_in_tensor(total_loss,"total_loss")
            total_loss.backward()


            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    alpha_mask = (gaussian_model.get_opacity()
                                  < self.pruning_thre).squeeze()
                    # nan_mask= torch.isnan(gaussian_model.get_scaling()).any(dim=1)
                    # gaussian_model.prune_points(alpha_mask&nan_mask)
                    gaussian_model.prune_points(alpha_mask)

                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step() 
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict

    def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
                    pts: np.ndarray, filter_cloud: bool) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        gaussian_points = gaussian_model.get_xyz()
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics) #视锥体8个角的世界坐标
        reused_pts_ids = compute_frustum_point_ids( #视锥体裁减
            gaussian_points, np2torch(camera_frustum_corners), device="cuda")
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device="cuda")
        new_pts_ids = torch2np(new_pts_ids)
        if new_pts_ids.shape[0] > 0:
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)#前三位坐标，后三是颜色
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0) #remove_statistical_outlier是用来移除统计学离群点（离均值远的点）
            gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False #为什么这里的颜色分量的梯度设置为了False？？？ 3.2节：我们直接优化RGB颜色，不使用球谐函数来加速优化。什么叫直接优化颜色？
        gaussian_model._features_rest.requires_grad = False
        print("Gaussian model size", gaussian_model.get_size())
        return new_pts_ids.shape[0]
    
    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """

        _, gt_color, gt_depth, _ , _, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        color_transform = torchvision.transforms.ToTensor()
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)}

        seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        pts = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)

        filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap

        new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud)

        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations
        start_time = time.time()
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)#将当前帧（由 frame_id 和 keyframe 组成的元组）添加到已有的关键帧列表，插入的顺序是按照加号的顺序来的
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe)) #添加新关键帧到该子图

        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
            psnr_value = calc_psnr(image_vis, keyframe["color"]).mean().item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")
            self.logger.vis_mapping_iteration(
                frame_id, max_iterations,
                image_vis.clone().detach().permute(1, 2, 0),
                depth_vis.clone().detach().permute(1, 2, 0),
                keyframe["color"].permute(1, 2, 0),
                keyframe["depth"].unsqueeze(-1),
                seeding_mask=seeding_mask)

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/max_iterations, opt_dict,False)
        return opt_dict
    

    def map_intrinsic(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool) -> dict:
        _, gt_color, gt_depth, _, gt_lab, normal_pseudo_cam= self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        tensor_transform = torchvision.transforms.ToTensor()
        #cal gt lab
        # gt_lab = rgb2lab(np2torch(gt_color,"cuda")/255)#[H,W,3],这个np2torch会把img从[3,H,W]->[H,W,3],而这正是rgb2lab需要的

        #cal pseudo normal,计算法线需要当前帧的c2w，按理说没法在加载数据集时运算
        # normal_pseudo_cam = -depth_to_normals(tensor_transform(gt_depth)[None].cuda(), tensor_transform(self.dataset.intrinsics.astype(np.float32)).cuda())[0]#(B,3,H,W,)三个rgb通道存储的该像素法线的方向向量，负号调整方向
        R = tensor_transform(estimate_c2w[:3, :3]).cuda()
        _, H, W = normal_pseudo_cam.shape
        pseudo_normal = (R @ torch.from_numpy(normal_pseudo_cam).float().cuda().reshape(3, -1)).reshape(3, H, W)

        keyframe = {
            "color": tensor_transform(gt_color).cuda(), #[3,H,W]
            "depth": np2torch(gt_depth, device="cuda"),#[H,W]
            "render_settings": get_render_settings_intrinsic(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c),
            "normal":pseudo_normal,#[3,H,W]
            "lab":tensor_transform(gt_lab).cuda(),#[3,H,W],tensor_transform的输出是[3,H,W]
            }
        seeding_mask = self.compute_seeding_mask_intrinsic(gaussian_model, keyframe, is_new_submap)
        pts = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)

        filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap

        new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud)

        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations
        start_time = time.time()
        opt_dict = self.optimize_submap_intrinsic([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe))

        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg = render_gaussian_model_intrinsic(gaussian_model, keyframe["render_settings"])
            # image_vis= render_pkg["color"]
            intrinsic_vis = render_pkg["intrinsic"]
            # psnr_value = calc_psnr(image_vis, keyframe["color"]).mean().item()
            psnr_value = calc_psnr(intrinsic_vis, keyframe["color"]).mean().item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")

            gt_image= keyframe["color"]
            gt_depth= keyframe["depth"]
            gt_normal = keyframe["normal"]
            rendered_image= render_pkg["color"]
            rendered_depth = render_pkg["depth"]
            rendered_intrinsic=render_pkg["intrinsic"]
            rendered_reflectance = render_pkg["reflectance"]
            rendered_shading = render_pkg["shading"]
            rendered_offset = render_pkg["offset"]
            rendered_residual = render_pkg["residual"]
            visualization_list = [
                gt_image,
                visualize_depth(gt_depth.unsqueeze(0)),
                gt_normal * 0.5 + 0.5,
                visualize_depth(rendered_depth),
                rendered_image,
                rendered_intrinsic,
                rendered_reflectance,
                rendered_shading.repeat(3, 1, 1),
                rendered_residual,
                rendered_reflectance+rendered_offset,
                rendered_offset,
            ]
            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            # grid = make_grid(grid, nrow=3)

            _, last_two_levels = os.path.split(os.path.split(self.dataset.dataset_path)[0])
            _, last_level = os.path.split(self.dataset.dataset_path)
            if not os.path.exists(os.path.join('output',last_two_levels,last_level,'vit')):
                os.makedirs(os.path.join('output',last_two_levels,last_level,'vit'))
            save_image(grid, os.path.join('output',last_two_levels,last_level,'vit',f"{frame_id:06d}.png"))

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/max_iterations, opt_dict,True)
        return opt_dict
