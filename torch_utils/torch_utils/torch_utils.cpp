// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
at::Tensor cuda_weighted_filter_forward(at::Tensor input, at::Tensor weight, int64_t kernel_size, bool splat);
std::vector<at::Tensor> cuda_weighted_filter_backward(at::Tensor grad_out, at::Tensor input, at::Tensor weight, int64_t kernel_size, bool splat);


//////////////////////////////////////////////////////////////////////////////////
// C++ / Python interface
//////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x) AT_ASSERTM(x.type().scalarType() == at::ScalarType::Float, #x " must be contiguous")
#define CHECK_DIM(x) AT_ASSERTM(x.dim() == 4LL, #x " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_TENSOR_4D_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x); CHECK_DIM(x)

at::Tensor weighted_filter_forward(
    at::Tensor input,
    at::Tensor weights,
    int64_t kernel_size,
    bool splat
) {
    CHECK_TENSOR_4D_FLOAT(input);
    CHECK_TENSOR_4D_FLOAT(weights);
    AT_ASSERTM(weights.size(0) == input.size(0) && weights.size(2) == input.size(2) && weights.size(3) == input.size(3), "Input and weight tensors missmatch");
    AT_ASSERTM(weights.size(1) == kernel_size * kernel_size, "Weight tensors and kernel size missmatch");
    AT_ASSERTM(kernel_size % 2 == 1, "Kernel size must be odd");

    return cuda_weighted_filter_forward(input, weights, kernel_size, splat);
}

std::vector<at::Tensor> weighted_filter_backward(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor weights,
    int64_t kernel_size,
    bool splat
) {
    CHECK_TENSOR_4D_FLOAT(grad_out);
    CHECK_TENSOR_4D_FLOAT(input);
    CHECK_TENSOR_4D_FLOAT(weights);
    AT_ASSERTM(grad_out.size(0) == input.size(0) && grad_out.size(2) == input.size(2) && grad_out.size(3) == input.size(3), "Input and gradient tensors missmatch");
    AT_ASSERTM(weights.size(0) == input.size(0) && weights.size(2) == input.size(2) && weights.size(3) == input.size(3), "Input and weight tensors missmatch");
    AT_ASSERTM(weights.size(1) == kernel_size * kernel_size, "Weight tensors and kernel size missmatch");
    AT_ASSERTM(kernel_size % 2 == 1, "Kernel size must be odd");

    return cuda_weighted_filter_backward(grad_out, input, weights, kernel_size, splat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_filter_forward", &weighted_filter_forward, "weighted_filter_forward");
    m.def("weighted_filter_backward", &weighted_filter_backward, "weighted_filter_backward");
}
