# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch_utils_cpp

class WeightedFilterFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weights, kernel_size, splat):
		# Store input activations and weights for back propagation pass
		ctx.save_for_backward(input, weights)

		# Store kernel size
		assert not hasattr(ctx, '_weighted_kernel_size') or ctx._weighted_kernel_size is None
		ctx._weighted_kernel_size = kernel_size
		ctx._weighted_splat = splat

		# Evaluate convolution
		return torch_utils_cpp.weighted_filter_forward(input, weights, kernel_size, splat)

	@staticmethod
	def backward(ctx, grad_out):
		grad_input, grad_weights = torch_utils_cpp.weighted_filter_backward(grad_out.contiguous(), *ctx.saved_variables, ctx._weighted_kernel_size, ctx._weighted_splat)
		return grad_input, grad_weights, None, None


class WeightedFilter(torch.nn.Module):
	def __init__(self, channels, kernel_size, bias=True, splat=False):
		super(WeightedFilter, self).__init__()
		self.in_channels  = channels
		self.out_channels = channels
		self.kernel_size  = kernel_size
		self.splat        = splat

		if bias:
			self.bias     = torch.nn.Parameter(torch.Tensor(self.out_channels))

	def forward(self, input, weight):
		bilat = WeightedFilterFunction.apply(input, weight, self.kernel_size, self.splat)
		if hasattr(self, 'bias'):
			return bilat + self.bias.view(1, self.out_channels, 1, 1)
		else:
			return bilat
