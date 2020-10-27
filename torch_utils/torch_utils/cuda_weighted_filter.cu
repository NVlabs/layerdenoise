// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>

#define WEIGHTS_TILE_SIZE 64

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static dim3 compute_blocks(dim3 size, dim3 threads)
{
    return dim3((size.x + threads.x - 1) / threads.x, (size.y + threads.y - 1) / threads.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward pass
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_weighted_filter_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int32_t batch_size,
    int32_t in_channels,
    int32_t height,
    int32_t width,
    int32_t filter_h,
    int32_t filter_w,
    bool splat)
{
    const int32_t ch          = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel_index >= height * width || ch >= in_channels)
        return;

    const int32_t py = pixel_index / width;
    const int32_t px = pixel_index % width;

    float result = 0.0f;
    for (int32_t fy = 0; fy < filter_h; ++fy)
    {
        for (int32_t fx = 0; fx < filter_w; ++fx)
        {
            // Compute tap coordinates, used for input activations and bilateral guides
            int32_t y = py + fy - (filter_h - 1) / 2;
            int32_t x = px + fx - (filter_w - 1) / 2;

            if (y < 0 || x < 0 || y >= height || x >= width)
                continue;

            // Filter using custom weight, use either gathering or splatting (scatter)
            if (splat)
                result += input[ch*height*width + y * width + x] * weight[((filter_h - fy - 1)*filter_w + (filter_w - fx - 1))*height*width + y*width + x];     // Splatting
            else
                result += input[ch*height*width + y*width + x] * weight[(fy*filter_w + fx)*height*width + py*width + px];   // Gathering
        }
    }
    output[ch*height*width + pixel_index] = result;
}

at::Tensor cuda_weighted_filter_forward(at::Tensor input, at::Tensor weight, int64_t kernel_size, bool splat)
{
    // Get tensor shapes
    at::IntList input_shape = input.sizes();
    at::IntList weight_shape = weight.sizes();
    at::IntList output_shape = input_shape;

    // Initialize output tensor to zero
    at::Tensor output = at::zeros(output_shape, input.options());

    // Setup dimensions for cuda kernel
    dim3 threads = dim3(32, 4);
    dim3 size    = dim3(input_shape[2] * input_shape[3], input_shape[1]);  // #pixels, out_channels = in_channels
    dim3 blocks  = compute_blocks(size, threads);

    // Invoke separate cuda kernel for each batch
    for (int64_t batch = 0; batch < input_shape[0]; ++batch)
    {
        cuda_weighted_filter_forward_kernel <<<blocks, threads>>> (
            input.data<float>()  + batch * input_shape[1] * input_shape[2] * input_shape[3],
            weight.data<float>() + batch * weight_shape[1] * weight_shape[2] * weight_shape[3],
            output.data<float>() + batch * output_shape[1] * output_shape[2] * output_shape[3],
            (int32_t)input_shape[0],  // batch_size
            (int32_t)input_shape[1],  // in_channels
            (int32_t)input_shape[2],  // height
            (int32_t)input_shape[3],  // width
            (int32_t)kernel_size,     // filter_h
            (int32_t)kernel_size,     // filter_w
            splat                     // splatting vs gather
            );
    }

    // Return result
    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward pass
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_weighted_filter_backward_kernel_activations(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_input,
    int32_t batch_size,
    int32_t in_channels,
    int32_t height,
    int32_t width,
    int32_t filter_h,
    int32_t filter_w,
    bool splat)
{
    const int32_t ch          = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel_index >= height * width || ch >= in_channels)
        return;

    const int32_t py = pixel_index / width;
    const int32_t px = pixel_index % width;

    float result = 0.0f;
    for (int32_t fy = 0; fy < filter_h; ++fy)
    {
        for (int32_t fx = 0; fx < filter_w; ++fx)
        {
            // Gradient and guide coordinates, regular unflipped. This probably wont work with even sized filters
            int32_t y = py + fy - (filter_h - 1) / 2;
            int32_t x = px + fx - (filter_w - 1) / 2;

            // Check for out-of-bounds access
            if (y < 0 || x < 0 || y >= height || x >= width)
                continue;

            // Compute activation derivative
            if (splat)
                result += grad_out[ch*height*width + y * width + x] * weight[(fy*filter_w + fx)*height*width + py * width + px];
            else
                result += grad_out[ch*height*width + y*width + x] * weight[((filter_h-fy-1)*filter_w + (filter_w-fx-1))*height*width + y*width + x];
        }
    }
    grad_input[ch*height*width + pixel_index] = result;
}

__global__ void cuda_weighted_filter_backward_kernel_weights(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_weight,
    int32_t batch_size,
    int32_t in_channels,
    int32_t height,
    int32_t width,
    int32_t filter_h,
    int32_t filter_w,
    bool splat)
{
    const int32_t weight_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t pixel_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel_index >= height * width || weight_index >= filter_h*filter_w)
        return;

    // Compute pixel coordinate
    const int32_t py = pixel_index / width;
    const int32_t px = pixel_index % width;

    // Compute tap/weight coordinate
    const int32_t fy = weight_index / filter_w;
    const int32_t fx = weight_index % filter_w;

    // Compute gradient, use zero if tap points to outside image region
    float result = 0.0f;
    if (splat)
    {
        // Compute tap offset in image space
        int32_t y = py + (fy - (filter_h - 1) / 2);
        int32_t x = px + (fx - (filter_w - 1) / 2);

        if (y >= 0 && x >= 0 && y < height && x < width)
        {
            for (int32_t ch = 0; ch < in_channels; ++ch)
            {
                // Result based on output gradient at pixel coordinate and input activation
               result += grad_out[ch*height*width + y*width + x] * input[ch*height*width + py*width + px];
            }
        }
    }
    else
    {
        // Compute tap offset in image space
        int32_t y = py + fy - (filter_h - 1) / 2;
        int32_t x = px + fx - (filter_w - 1) / 2;

        if (y >= 0 && x >= 0 && y < height && x < width)
        {
            for (int32_t ch = 0; ch < in_channels; ++ch)
            {
                // Result based on output gradient at pixel coordinate and input activation
                result += grad_out[ch*height*width + py*width + px] * input[ch*height*width + y*width + x];
            }
        }
    }

    grad_weight[weight_index*height*width + pixel_index] = result;
}


std::vector<at::Tensor> cuda_weighted_filter_backward(at::Tensor grad_out, at::Tensor input, at::Tensor weight, int64_t kernel_size, bool splat)
{
    // Get tensor shapes
    at::IntList input_shape  = input.sizes();
    at::IntList weight_shape = weight.sizes();
    at::IntList output_shape = grad_out.sizes();

    // Initialize output gradient tensors to zero
    at::Tensor input_grad  = at::zeros_like(input);
    at::Tensor weight_grad = at::zeros_like(weight);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Gradients for input activations
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        // Setup dimensions for cuda kernel
        dim3 threads = dim3(128, 4);
        dim3 size = dim3(input_shape[2] * input_shape[3], input_shape[1]);  // #pixels, #out_channels = in_channels
        dim3 blocks = compute_blocks(size, threads);

        // Invoke separate cuda kernel for each batch
        for (int64_t batch = 0; batch < input_shape[0]; ++batch)
        {
            cuda_weighted_filter_backward_kernel_activations << <blocks, threads >> > (
                grad_out.data<float>() + batch * output_shape[1] * output_shape[2] * output_shape[3],
                weight.data<float>() + batch * weight_shape[1] * weight_shape[2] * weight_shape[3],
                input_grad.data<float>() + batch * input_shape[1] * input_shape[2] * input_shape[3],
                (int32_t)input_shape[0],    // batch_size
                (int32_t)input_shape[1],    // in_channels
                (int32_t)input_shape[2],    // height
                (int32_t)input_shape[3],    // width
                (int32_t)kernel_size,       // filter_h
                (int32_t)kernel_size,       // filter_w
                splat                       // splatting vs gather
                );
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Gradients for weights
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        // Setup dimensions for cuda kernel
        dim3 threads = dim3(128, 4);
        dim3 size = dim3(weight_shape[2] * weight_shape[3], weight_shape[1]);  // #pixels, #weights
        dim3 blocks = compute_blocks(size, threads);

        // Invoke separate cuda kernel for each batch
        for (int64_t batch = 0; batch < input_shape[0]; ++batch)
        {
            cuda_weighted_filter_backward_kernel_weights << <blocks, threads >> > (
                grad_out.data<float>() + batch * output_shape[1] * output_shape[2] * output_shape[3],
                input.data<float>() + batch * input_shape[1] * input_shape[2] * input_shape[3],
                weight_grad.data<float>() + batch * weight_shape[1] * weight_shape[2] * weight_shape[3],
                (int32_t)input_shape[0],    // batch_size
                (int32_t)input_shape[1],    // in_channels
                (int32_t)input_shape[2],    // height
                (int32_t)input_shape[3],    // width
                (int32_t)kernel_size,       // filter_h
                (int32_t)kernel_size,       // filter_w
                splat                       // splatting vs gather
                );
        }
    }

    return { input_grad, weight_grad };
}
