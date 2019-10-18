//------------------------------------------------------------------------------
// Copyright (c) 2019 by contributors. All Rights Reserved.
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
//------------------------------------------------------------------------------

/*
This file defines the Youtube DNN refer to paper <<Deep Neural Networks for YouTube Recommendations>> class.
*/

#ifndef XLEARN_LOSS_YOUTUBEDNN_SCORE_H_
#define XLEARN_LOSS_YOUTUBEDNN_SCORE_H_

#include "src/base/common.h"
#include "src/data/model_parameters.h"
#include "src/score/score_function.h"

namespace xLearn {


class YOUTUBEDNNScore : public Score {
 public:
  // Constructor and Desstructor
  YOUTUBEDNNScore() { }
  ~YOUTUBEDNNScore() { }

  // Given one exmaple and current model, this method
  // returns the Youtube DNN score.
  real_t CalcScore(const SparseRow* row,
                   Model& model,
                   index_t thread_i,
                   real_t norm = 1.0);

  // Calculate gradient
  // model parameters.
  void CalcGrad(const SparseRow* row,
                Model& model,
                index_t thread_i,
                real_t pg,
                real_t norm = 1.0);

  // update
  void UpDate(Model& model,index_t thread_i);

 protected:
  // Calculate gradient and update model using sgd
  void calc_grad_sgd(const SparseRow* row,
                     Model& model,
                     index_t thread_i,
                     real_t pg,
                     real_t norm = 1.0);

  real_t relu(real_t input);



 private:
  real_t* comp_res = nullptr;
  real_t* comp_z_lt_zero = nullptr;
  real_t* comp_z_gt_zero = nullptr;

 private:
  DISALLOW_COPY_AND_ASSIGN(YOUTUBEDNNScore);
};

} // namespace xLearn

#endif // XLEARN_LOSS_YOUTUBEDNN_SCORE_H_
