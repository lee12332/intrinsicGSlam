import os
import random

import numpy as np
import open3d as o3d
import torch

from src.entities.gaussian_model import GaussianModel
# from gaussian_rasterizer import GaussianRasterizationSettings as vanillaRasterizationSettings
# from gaussian_rasterizer import GaussianRasterizer as vanillaRasterizer

from src.intrinsic.r3dg_rasterization import GaussianRasterizationSettings as intrinsicRasterizationSettings
from src.intrinsic.r3dg_rasterization import GaussianRasterizer as intrinsicRasterizer
from src.intrinsic.utils_intrinsic import intrinsic_composition
from src.utils.gaussian_model_utils import eval_sh
 

def setup_seed(seed: int) -> None:
    """ Sets the seed for generating random numbers to ensure reproducibility across multiple runs.
    Args:
        seed: The seed value to set for random number generators in torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    """Sends all tensors in a dictionary to a specified device.
    Args:
        dict: The dictionary containing tensors.
        device: The device to send the tensors to. Defaults to 'cpu'.
    Returns:
        The dictionary with all tensors sent to the specified device.
    """
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            dict[k] = v.to(device)
    return dict


# def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
#     """
#     Constructs and returns a GaussianRasterizationSettings object for rendering,
#     configured with given camera parameters.

#     Args:
#         width (int): The width of the image.
#         height (int): The height of the image.
#         intrinsic (array): 3*3, Intrinsic camera matrix.
#         w2c (array): World to camera transformation matrix.
#         near (float, optional): The near plane for the camera. Defaults to 0.01.
#         far (float, optional): The far plane for the camera. Defaults to 100.

#     Returns:
#         GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
#     """
#     fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
#                                                   1], intrinsics[0, 2], intrinsics[1, 2]
#     w2c = torch.tensor(w2c).cuda().float()
#     cam_center = torch.inverse(w2c)[:3, 3]
#     viewmatrix = w2c.transpose(0, 1)
#     opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
#                                 [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
#                                 [0.0, 0.0, far /
#                                     (far - near), -(far * near) / (far - near)],
#                                 [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
#     full_proj_matrix = viewmatrix.unsqueeze(
#         0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
#     return vanillaRasterizationSettings(
#         image_height=h,
#         image_width=w,
#         tanfovx=w / (2 * fx),
#         tanfovy=h / (2 * fy),
#         bg=torch.tensor([0, 0, 0], device='cuda').float(),
#         scale_modifier=1.0,
#         viewmatrix=viewmatrix,
#         projmatrix=full_proj_matrix,
#         sh_degree=sh_degree,
#         campos=cam_center,
#         prefiltered=False,
#         debug=False)

def get_render_settings_intrinsic(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /(far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return intrinsicRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        cx=cx,
        cy=cy,
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        backward_geometry=True,
        # computer_pseudo_normal=True,
        computer_pseudo_normal=False,
        debug=False
    )


# def render_gaussian_model(gaussian_model, render_settings,
#                           override_means_3d=None, override_means_2d=None,
#                           override_scales=None, override_rotations=None,
#                           override_opacities=None, override_colors=None):
#     """
#     Renders a Gaussian model with specified rendering settings, allowing for
#     optional overrides of various model parameters.
#     track的时候是重载了means_3d和rotation，其他时候都没有重载

#     Args:
#         gaussian_model: A Gaussian model object that provides methods to get
#             various properties like xyz coordinates, opacity, features, etc.
#         render_settings: Configuration settings for the GaussianRasterizer.
#         override_means_3d (Optional): If provided, these values will override
#             the 3D mean values from the Gaussian model.
#         override_means_2d (Optional): If provided, these values will override
#             the 2D mean values. Defaults to zeros if not provided.
#         override_scales (Optional): If provided, these values will override the
#             scale values from the Gaussian model.
#         override_rotations (Optional): If provided, these values will override
#             the rotation values from the Gaussian model.
#         override_opacities (Optional): If provided, these values will override
#             the opacity values from the Gaussian model.
#         override_colors (Optional): If provided, these values will override the
#             color values from the Gaussian model.
#     Returns:
#         A dictionary containing the rendered color, depth, radii, and 2D means
#         of the Gaussian model. The keys of this dictionary are 'color', 'depth',
#         'radii', and 'means2D', each mapping to their respective rendered values.
#     """
#     renderer = vanillaRasterizer(raster_settings=render_settings) #这是对GaussianRasterizer构造函数的调用raster_settings是构造函数的参数

#     if override_means_3d is None:
#         means3D = gaussian_model.get_xyz()
#     else:
#         means3D = override_means_3d

#     if override_means_2d is None:
#         means2D = torch.zeros_like(
#             means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
#         means2D.retain_grad()
#     else:
#         means2D = override_means_2d

#     if override_opacities is None:
#         opacities = gaussian_model.get_opacity()
#     else:
#         opacities = override_opacities

#     shs, colors_precomp = None, None
#     if override_colors is not None:
#         colors_precomp = override_colors
#     else:
#         shs = gaussian_model.get_features()

#     render_args = {
#         "means3D": means3D,
#         "means2D": means2D,
#         "opacities": opacities,
#         "colors_precomp": colors_precomp,
#         "shs": shs,
#         "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
#         "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
#         "cov3D_precomp": None
#     }
#     color, depth, alpha, radii = renderer(**render_args) #这是对GaussianRasterizer的forward函数的调用，render_args是forward函数的参数


#     return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}

def render_gaussian_model_intrinsic(gaussian_model: GaussianModel, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):

    renderer = intrinsicRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    #intrinsic attribute
    reflectance = gaussian_model.get_reflectance() # [N, 3]
    shading = gaussian_model.get_shading() # [N, 1]
    residual = gaussian_model.get_residual() # [N, 16, 3]
    offset = gaussian_model.get_offset() # [N, 3]
    viewdirs = torch.nn.functional.normalize(render_settings.campos - means3D, dim=-1) #track时应该是上一帧的campos和变换后的pts
    #intrinsic=(reflect+delta）*shading+residual
    intrinsic_color = intrinsic_composition(reflectance, shading, offset, residual, viewdirs)
    intrinsic_list = [intrinsic_color, reflectance, shading, offset]
    ch_list = [3, 3, 1, 3]
    #residual part
    deg = int(np.sqrt(residual.shape[1]) - 1)
    residual_shs_view = residual.transpose(1, 2).view(-1, 3, residual.shape[1]) # [N, 3, 16]
    residual_view = eval_sh(deg, residual_shs_view, viewdirs) # [N, 3]
    residual_view = torch.clamp_min(residual_view + 0.5, 0.0)
    #list and shape
    intrinsic_list.append(residual_view)#feature_list = [intrinsic_color, reflectance, shading, offset, residual_view]
    ch_list.append(residual_view.shape[-1])
    intrinsic_features = torch.cat(intrinsic_list, dim=-1)#[N, 3 + 3 + 1 + 3 + 3]

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacities": gaussian_model.get_opacity() if override_opacities is None else override_opacities,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None,
        "features": intrinsic_features
    }
    (num_rendered, num_contrib, rendered_color, rendered_opacity, rendered_depth,
        rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = renderer(**render_args)

    rendered_list = rendered_feature.split(ch_list, dim=0)
    rendered_intrinsic = rendered_list[0]
    rendered_reflectance = rendered_list[1]
    rendered_shading = rendered_list[2]
    rendered_offset = rendered_list[3]
    rendered_residual = rendered_list[-1]


    return {"color": rendered_color, 
            "depth": rendered_depth, 
            "radii": radii, 
            "alpha": rendered_opacity,
            "intrinsic":rendered_intrinsic,
            "reflectance":rendered_reflectance,
            "shading":rendered_shading,
            "offset":rendered_offset,
            "residual":rendered_residual}


def batch_search_faiss(indexer, query_points, k):
    """
    Perform a batch search on a IndexIVFFlat indexer to circumvent the search size limit of 65535.

    Args:
        indexer: The FAISS indexer object.
        query_points: A tensor of query points.
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (torch.Tensor): The distances of the nearest neighbors.
        ids (torch.Tensor): The indices of the nearest neighbors.
    """
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float(), k)
        distances_list.append(distance.clone())
        ids_list.append(id.clone())
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids
