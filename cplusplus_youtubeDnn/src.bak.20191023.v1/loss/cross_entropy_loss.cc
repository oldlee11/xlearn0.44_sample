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
#include <atomic>


namespace xLearn {


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


void CrossEntropyLoss::Init_Atomic(){
    update_lock_=0;
    isUpDateing_= false;
    assert(threadNumber_<=16);
    isBPing_.reserve(threadNumber_);
    // is error the p of the tmp is same in each vector elements
//    for(index_t i=0;i<threadNumber_;i++){
//        std::atomic<bool> tmp(false);
//        isBPing_.push_back(&tmp);
//    }
    if(threadNumber_>=1){
        isBPing_Thread_0_=false;
        isBPing_.push_back(&isBPing_Thread_0_);
    }
    if(threadNumber_>=2){
        isBPing_Thread_1_=false;
        isBPing_.push_back(&isBPing_Thread_1_);
    }
    if(threadNumber_>=3){
        isBPing_Thread_2_=false;
        isBPing_.push_back(&isBPing_Thread_2_);
    }
    if(threadNumber_>=4){
        isBPing_Thread_3_=false;
        isBPing_.push_back(&isBPing_Thread_3_);
    }
    if(threadNumber_>=5){
        isBPing_Thread_4_=false;
        isBPing_.push_back(&isBPing_Thread_4_);
    }
    if(threadNumber_>=6){
        isBPing_Thread_5_=false;
        isBPing_.push_back(&isBPing_Thread_5_);
    }
    if(threadNumber_>=7){
        isBPing_Thread_6_=false;
        isBPing_.push_back(&isBPing_Thread_6_);
    }
    if(threadNumber_>=8){
        isBPing_Thread_7_=false;
        isBPing_.push_back(&isBPing_Thread_7_);
    }
    if(threadNumber_>=9){
        isBPing_Thread_8_=false;
        isBPing_.push_back(&isBPing_Thread_8_);
    }
    if(threadNumber_>=10){
        isBPing_Thread_9_=false;
        isBPing_.push_back(&isBPing_Thread_9_);
    }
    if(threadNumber_>=11){
        isBPing_Thread_10_=false;
        isBPing_.push_back(&isBPing_Thread_10_);
    }
    if(threadNumber_>=12){
        isBPing_Thread_11_=false;
        isBPing_.push_back(&isBPing_Thread_11_);
    }
    if(threadNumber_>=13){
        isBPing_Thread_12_=false;
        isBPing_.push_back(&isBPing_Thread_12_);
    }
    if(threadNumber_>=14){
        isBPing_Thread_13_=false;
        isBPing_.push_back(&isBPing_Thread_13_);
    }
    if(threadNumber_>=15){
        isBPing_Thread_14_=false;
        isBPing_.push_back(&isBPing_Thread_14_);
    }
    if(threadNumber_>=16){
        isBPing_Thread_15_=false;
        isBPing_.push_back(&isBPing_Thread_15_);
    }

}


// Calculate loss in one thread.
static void ce_evalute_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  /*
   计算结果居然一样
   for (size_t i = start_idx; i < end_idx; ++i) {
    real_t y = (*label)[i] > 0 ? 1.0 : -1.0;
    (*tmp_sum) += log1p(exp(-y*(*pred)[i]));
  }
  */
  real_t pred2=0.0;
  for (size_t i = start_idx; i < end_idx; ++i) {
      real_t y = ((*label)[i] > 0.5) ? 1.0 : 0.0;
      pred2=sigmod((*pred)[i]);
      pred2 =cli_value(pred2);
      (*tmp_sum) += -1.0 * ( (y > 0.5) ? log(pred2) : log(1.0-pred2) );
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


static void ce_gradient_thread(const DMatrix* matrix,
                               Model* model,
                               index_t thread_i,
                               index_t threads,
                               Score* score_func,
                               bool is_norm,
                               real_t* sum,
                               size_t start_idx,
                               size_t end_idx,
                               real_t batch_size,
                               std::atomic<bool> *p_isUpDateing_,
                               std::atomic<index_t> *p_update_lock_,
                               std::vector<std::atomic<bool>*> *p_isBPing_
) {
    CHECK_GE(end_idx, start_idx);
    *sum = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        /*
         * waitting until
         * there is no threads running update function
         * */
        while(true){
            if(!(*p_isUpDateing_)){
                break;
            }
        }
        // std::cout << "thread_" << thread_i << " bp begin" << std::endl;
        *(*p_isBPing_)[thread_i] =true;  // running bp begin
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
        // std::cout << "thread_" << thread_i << " bp end" << std::endl;
        *(*p_isBPing_)[thread_i]=false;// running bp end


        // update
        if(*p_update_lock_==thread_i){
            if( (*(model->GetFulllayer_change_num()+thread_i)) > batch_size){ // 8.0 means batch_size
                // std::cout << "thread_" << thread_i << " update begin" << std::endl;
                *p_isUpDateing_=true;  // running update begin
                /*
                 * waitting until
                 * there is no threads running bp function
                 * */
                bool tmp_bool=true;  // true=没有一个线程在运行bp
                while(true){
                    tmp_bool=true;
                    for(index_t tmp_i =0; tmp_i<threads ; tmp_i++){
                        if(*(*p_isBPing_)[tmp_i]){ // 如果有任意一个线程在运行bp 则 重新等待检查
                            tmp_bool=false;
                            break;
                        }
                    }
                    if(tmp_bool){
                        break;
                    }
                }
                // std::cout << "thread_" << thread_i << " update real begin" << std::endl;
                // score_func->UpDate(*model, thread_i);
                score_func->UpDate_AllThreads(*model);
                // std::cout << "thread_" << thread_i << " update end" << std::endl;
                *p_isUpDateing_=false; // running update end
            }
            *p_update_lock_ = (thread_i==(threads - 1)) ? 0 : (thread_i+1);
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
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
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
                             end_idx,
                             batch_size_,
                             &isUpDateing_,
                             &update_lock_,
                             &isBPing_
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
