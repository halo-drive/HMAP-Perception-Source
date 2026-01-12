# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Updated to use spconv 2.x
# This file now acts as a wrapper to import from the official spconv 2.x package
# instead of the bundled spconv 1.x implementation

import spconv.pytorch as spconv

# Import from spconv 2.x
from spconv.pytorch import (
    SparseConv2d,
    SparseConv3d,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SubMConv2d,
    SubMConv3d,
    SparseModule,
    SparseSequential,
    SparseMaxPool2d,
    SparseMaxPool3d,
    SparseConvTensor,
)

# Register spconv 2.x classes with mmcv's CONV_LAYERS so build_conv_layer can find them
from mmcv.cnn import CONV_LAYERS

# Register the spconv 2.x classes by their class names
# This allows build_conv_layer to find them when config specifies type: "SubMConv3d"
CONV_LAYERS.register_module(name="SparseConv3d")(SparseConv3d)
CONV_LAYERS.register_module(name="SubMConv3d")(SubMConv3d)
CONV_LAYERS.register_module(name="SparseConv2d")(SparseConv2d)
CONV_LAYERS.register_module(name="SubMConv2d")(SubMConv2d)
CONV_LAYERS.register_module(name="SparseConvTranspose3d")(SparseConvTranspose3d)
CONV_LAYERS.register_module(name="SparseInverseConv3d")(SparseInverseConv3d)

__all__ = [
    "SparseConv2d",
    "SparseConv3d",
    "SubMConv2d",
    "SubMConv3d",
    "SparseConvTranspose2d",
    "SparseConvTranspose3d",
    "SparseInverseConv2d",
    "SparseInverseConv3d",
    "SparseModule",
    "SparseSequential",
    "SparseMaxPool2d",
    "SparseMaxPool3d",
    "SparseConvTensor",
    "spconv",
]
