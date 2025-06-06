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

#include "ultra_infer/vision/common/processors/base.h"

namespace ultra_infer {
namespace vision {

/*! @brief Processor for Limit images by short edge with given parameters.
 */
class LimitShort : public Processor {
public:
  explicit LimitShort(int max_short = -1, int min_short = -1, int interp = 1) {
    max_short_ = max_short;
    min_short_ = min_short;
    interp_ = interp;
  }

  // Limit the short edge of image.
  // If the short edge is larger than max_short_, resize the short edge
  // to max_short_, while scale the long edge proportionally.
  // If the short edge is smaller than min_short_, resize the short edge
  // to min_short_, while scale the long edge proportionally.
  bool ImplByOpenCV(Mat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
  std::string Name() { return "LimitShort"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] max_short target size of short edge
   * \param[in] min_short target size of short edge
   * \param[in] interp interpolation method, default is 1
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(Mat *mat, int max_short = -1, int min_short = -1,
                  int interp = 1, ProcLib lib = ProcLib::DEFAULT);
  int GetMaxShort() const { return max_short_; }

private:
  int max_short_;
  int min_short_;
  int interp_;
};
} // namespace vision
} // namespace ultra_infer
