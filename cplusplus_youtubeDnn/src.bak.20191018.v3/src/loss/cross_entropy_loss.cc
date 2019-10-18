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
This file is the implementation of CrossEntropyLoss class.
*/

#include "src/loss/cross_entropy_loss.h"

#include <thread>
#include<atomic>


namespace xLearn {
/* means which thread shoud be run update func    c++ 原子锁
* update_lock_ in [0,threadNumber_-1]
* and loop to run update in each thread
*/
std::atomic<index_t> update_lock_;

// Calculate loss in one thread.
static void ce_evalute_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t y = (*label)[i] > 0 ? 1.0 : -1.0;
    (*tmp_sum) += log1p(exp(-y*(*pred)[i]));
  }
}

//------------------------------------------------------------------------------
// Calculate loss in multi-thread:
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
                               const std::vector<real_t>& label) {
  update_lock_=0;
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  total_example_ += pred.size();
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(ce_evalute_thread,
                             &pred,
                             &label,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
  // Accumulate loss
  for (size_t i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

/* Calculate gradient in one thread.
   https://www.cnblogs.com/nowgood/p/sigmoidcrossentropy.html
*/
inline real_t sigmod(real_t z){
    return 1.0/(1.0+exp(-1.0 * z));
}
inline real_t cli_value(real_t input){
    real_t input2 = (input <= 0.00001) ? 0.00001 : input ;
    return  (input2 >= 0.99999) ? 0.99999 : input2 ;
}
static void ce_gradient_thread(const DMatrix* matrix,
                               Model* model,
                               index_t thread_i,
                               index_t threads,
                               Score* score_func,
                               bool is_norm,
                               real_t* sum,
                               size_t start_idx,
                               size_t end_idx
) {
    CHECK_GE(end_idx, start_idx);
    *sum = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        SparseRow* row = matrix->row[i];
        real_t norm = is_norm ? matrix->norm[i] : 1.0;
        real_t pred = score_func->CalcScore(row, *model, thread_i,norm);

        // debug
//        if(std::isnan(*sum) == 1){// pred =nan
//            std::cout<<"isnan:"<< *sum << std::endl;
//        }

        pred=sigmod(pred);

        // debug
//        if(std::isnan(*sum) == 1){// pred =nan
//            std::cout<<"isnan:"<< *sum << std::endl;
//        }

        pred =cli_value(pred);

        // partial gradient
        real_t label = (matrix->Y[i] > 0.5) ? 1.0 : 0.0;
        *sum += -1.0 * ( (label > 0.5) ? log(pred) : log(1.0-pred) );

        // debug
//        if(std::isinf(*sum) == 1){
//            std::cout<<"isinf:"<< *sum << std::endl;
//        }
//        if(std::isnan(*sum) == 1){// pred =nan
//            std::cout<<"isnan:"<< *sum << std::endl;
//        }

        real_t pg = pred - label;
        // real gradient
        score_func->CalcGrad(row, *model, thread_i,pg, norm);

        // update
        if(update_lock_==thread_i){
            if( (*(model->GetFulllayer_change_num()+thread_i)) > 8.0){ // 8.0 means batch_size
                score_func->UpDate(*model, thread_i);
            }
            update_lock_ = (thread_i==(threads - 1)) ? 0 : (thread_i+1);
        }
    }
}



//------------------------------------------------------------------------------
// Calculate gradient in multi-thread
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::CalcGrad(const DMatrix* matrix,Model& model) {
  update_lock_=0;
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  // multi-thread training
  index_t count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    pool_->enqueue(std::bind(ce_gradient_thread,
                             matrix,
                             &model,
                             i,
                             count,
                             score_func_,
                             norm_,
                             &(sum[i]),
                             start_idx,
                             end_idx
    ));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
