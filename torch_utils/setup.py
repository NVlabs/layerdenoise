# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FILES = [
	'torch_utils/torch_utils.cpp', 
	'torch_utils/cuda_weighted_filter.cu',
]

setup(
	name='torch_utils',
	version='0.1',
    author="Jon Hasselgren",
    author_email="jhasselgren@nvidia.com",
    description="torch_utils - fast kernel evaluations",
    url="https://github.com/NVlabs/layerdenoise",
	install_requires=['torch'],
	packages=find_packages(exclude=['test*']),
	ext_modules=[CUDAExtension('torch_utils_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["torch_utils/weighted_filter"],
	cmdclass={
		'build_ext': BuildExtension
	},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
	python_requires='>=3.6',
)
