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

#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/utils/utils.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#ifndef NON_64_PLATFORM
#include "onnxruntime_cxx_api.h" // NOLINT

#ifdef WITH_GPU
#include "ultra_infer/runtime/backends/common/cuda/adaptive_pool2d_kernel.h"
#endif

namespace ultra_infer {
struct AdaptivePool2dKernel {
protected:
  std::string pooling_type_ = "avg";
  std::vector<int64_t> output_size_ = {};
  OrtApi ort_;
  void *compute_stream_;
  const char *provider_;

public:
  AdaptivePool2dKernel(OrtApi ort, const OrtKernelInfo *info,
                       const char *provider)
      : ort_(ort) {
    GetAttribute(info);
    provider_ = provider;
  }

  void GetAttribute(const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

  OrtStatusPtr ComputeV2(OrtKernelContext *context);

  void CpuAdaptivePool(const std::vector<int64_t> &input_size,
                       const std::vector<int64_t> &output_size,
                       const float *input_data, float *output_data);
};

struct AdaptivePool2dOp
    : Ort::CustomOpBase<AdaptivePool2dOp, AdaptivePool2dKernel> {
  explicit AdaptivePool2dOp(const char *provider) : provider_(provider) {}
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new AdaptivePool2dKernel(api, info, provider_);
  }

  OrtStatusPtr CreateKernelV2(OrtApi api, const OrtKernelInfo *info,
                              void **op_kernel) const {
    *op_kernel = new AdaptivePool2dKernel(api, info, provider_);
    return nullptr;
  }

  const char *GetName() const { return "AdaptivePool2d"; }

  size_t GetInputTypeCount() const { return 1; }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  const char *GetExecutionProviderType() const { return provider_; }

private:
  const char *provider_;
};

} // namespace ultra_infer

#endif
