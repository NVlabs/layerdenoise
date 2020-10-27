# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.10-py3
FROM $BASE_IMAGE

# Install torch_utils
COPY torch_utils /tmp/torch_utils/
RUN cd /tmp/torch_utils && python setup.py install
