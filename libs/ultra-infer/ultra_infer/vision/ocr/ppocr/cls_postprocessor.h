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
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
namespace vision {

namespace ocr {
/*! @brief Postprocessor object for Classifier serials model.
 */
class ULTRAINFER_DECL ClassifierPostprocessor {
public:
  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] cls_labels The output label results of classification model
   * \param[in] cls_scores The output score results of classification model
   * \return true if the postprocess succeeded, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<int32_t> *cls_labels, std::vector<float> *cls_scores);

  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<int32_t> *cls_labels, std::vector<float> *cls_scores,
           size_t start_index, size_t total_size);

  /// Set threshold for the classification postprocess, default is 0.9
  void SetClsThresh(float cls_thresh) { cls_thresh_ = cls_thresh; }

  /// Get threshold value of the classification postprocess.
  float GetClsThresh() const { return cls_thresh_; }

private:
  float cls_thresh_ = 0.9;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
