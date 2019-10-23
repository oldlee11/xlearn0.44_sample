//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file defines the CrossEntropyLoss class.
*/

#ifndef XLEARN_LOSS_CROSS_ENTROPY_LOSS_H_
#define XLEARN_LOSS_CROSS_ENTROPY_LOSS_H_

#include "src/base/common.h"
#include "src/loss/loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// CrossEntropyLoss is used for classification tasks, which
// has the following form:
// loss = sum_all_example(log(1.0+exp(-y*pred)))
//------------------------------------------------------------------------------
class CrossEntropyLoss : public Loss {
 public:
  // Constructor and Desstructor
  CrossEntropyLoss() { }
  ~CrossEntropyLoss() { }


  void Init_Atomic();


  // Given predictions and labels, accumulate cross-entropy loss.
  void Evalute(const std::vector<real_t>& pred,
               const std::vector<real_t>& label);

  // Given data sample and current model, calculate gradient
  // and update current model parameters.
  // This function will also accumulate the loss value.
  void CalcGrad(const DMatrix* data_matrix, Model& model);

  // Return current loss type.
  std::string loss_type() { return "log_loss"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(CrossEntropyLoss);

 protected:
    /*
     * 每个线程可以运行的逻辑分为bp(计算梯度变化)和update(把梯度变化更新到参数中)
     * 其中update必须只能有一个线程来运行,且在运行update过程中不能有线程运行bp
     * 运行update的线程的顺序是[0,threadNumber_-1] 依次循环运行
    */
    // 现在是否在运行update程序？
    std::atomic<bool> isUpDateing_;
    // 那个线程正在或即将运行update程序？ 保证只能有1个thread线程运行update
    std::atomic<index_t> update_lock_;
    // 每个线程是否在执行bp？
    std::vector<std::atomic<bool>*> isBPing_;
    // 原子操作 定义 有的扯 不能直接在vector中
    std::atomic<bool> isBPing_Thread_0_;
    std::atomic<bool> isBPing_Thread_1_;
    std::atomic<bool> isBPing_Thread_2_;
    std::atomic<bool> isBPing_Thread_3_;
    std::atomic<bool> isBPing_Thread_4_;
    std::atomic<bool> isBPing_Thread_5_;
    std::atomic<bool> isBPing_Thread_6_;
    std::atomic<bool> isBPing_Thread_7_;
    std::atomic<bool> isBPing_Thread_8_;
    std::atomic<bool> isBPing_Thread_9_;
    std::atomic<bool> isBPing_Thread_10_;
    std::atomic<bool> isBPing_Thread_11_;
    std::atomic<bool> isBPing_Thread_12_;
    std::atomic<bool> isBPing_Thread_13_;
    std::atomic<bool> isBPing_Thread_14_;
    std::atomic<bool> isBPing_Thread_15_;
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_CROSS_ENTROPY_LOSS_H_
