/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "render_equation.h"

//JIT即时编译链接python和cuda方式
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {//PyBind11 模块，名字为 TORCH_EXTENSION_NAME，对象是m
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);//m.def：将 C++函数绑定到 Python 中，使其可以在 Python 中调用
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("render_equation_forward", &RenderEquationForwardCUDA);
  m.def("render_equation_forward_complex", &RenderEquationForwardCUDA_complex);
  m.def("render_equation_backward", &RenderEquationBackwardCUDA);
  m.def("mark_visible", &markVisible);
}