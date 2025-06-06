// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "ultra_infer/vision/common/processors/cast.h"
#include "ultra_infer/vision/common/processors/center_crop.h"
#include "ultra_infer/vision/common/processors/color_space_convert.h"
#include "ultra_infer/vision/common/processors/convert.h"
#include "ultra_infer/vision/common/processors/convert_and_permute.h"
#include "ultra_infer/vision/common/processors/crop.h"
#include "ultra_infer/vision/common/processors/hwc2chw.h"
#include "ultra_infer/vision/common/processors/limit_by_stride.h"
#include "ultra_infer/vision/common/processors/limit_short.h"
#include "ultra_infer/vision/common/processors/normalize.h"
#include "ultra_infer/vision/common/processors/normalize_and_permute.h"
#include "ultra_infer/vision/common/processors/pad.h"
#include "ultra_infer/vision/common/processors/pad_to_size.h"
#include "ultra_infer/vision/common/processors/resize.h"
#include "ultra_infer/vision/common/processors/resize_by_short.h"
#include "ultra_infer/vision/common/processors/stride_pad.h"
#include "ultra_infer/vision/common/processors/warp_affine.h"
#include <unordered_set>

namespace ultra_infer {
namespace vision {

void FuseTransforms(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + Cast(Float) to Normalize
void FuseNormalizeCast(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + HWC2CHW to NormalizeAndPermute
void FuseNormalizeHWC2CHW(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + Color Convert
void FuseNormalizeColorConvert(
    std::vector<std::shared_ptr<Processor>> *processors);

} // namespace vision
} // namespace ultra_infer
