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
This file is the implementation of YOUTUBEDNNScore class.
*/

#include "src/score/youtubeDnn_score.h"
#include "src/base/math.h"

namespace xLearn {

inline real_t YOUTUBEDNNScore::relu(real_t input){
    return (input>0.0) ? input:0.0;
}


real_t YOUTUBEDNNScore::CalcScore(
        const SparseRow* row,
        Model& model,
        index_t thread_i,
        real_t norm) {
    // real_t sqrt_norm = sqrt(norm);
    index_t aligned_k = model.get_aligned_k();
    index_t num_feat = model.GetNumFeature();
    index_t num_fullLayer =model.GetNumFullLayerCell();


    /*********************************************************
     * mean elements in each filedi
     * >> sv_fileds
     *
     *                                filed0:element_sum                         filed1:element_mean
     *                                         /\                                          /\
     *                                         ||                                          ||
     *                                         ||                                          ||
     *                        ------------------------------------                         ||
     *                        /                 |                \                         ||
     *                       /                  |                 \                        ||
     *                embedding_0*value   embedding_1*value   embedding_2*value            ||
     *                     /                    |                   \                      ||
     *                    /                     |                    \                     ||
     *  input    filed0:index0:value   filed0:index1:value  filed0:index2:value   filed1:index3:value ....
     *  filed_i  must 0,1,<1,1,...>,2,<2,2,...>,3,<3,3,...>,...,n,<n,n,...>
     *                (filed0 means target rid ,we just need one )
     *********************************************************/
    index_t pre_f1 = 0 ;
    index_t _e = 0;
    for (SparseRow::const_iterator iter = row->begin(); iter != row->end(); ++iter) {
        index_t f1 = iter->field_id;
        index_t j1 = iter->feat_id;
        // To avoid unseen feature in Prediction
        if (j1 >= num_feat) {
            pre_f1 = f1;
            continue;
        }
        real_t v1 = iter->feat_val;
        // get feat j embedding
        real_t *w = model.GetEmbedding_v(j1);
        real_t _XMMv = v1 * norm;
        if (pre_f1 != f1) {
            _e += aligned_k;
        }
        for (index_t _d = 0; _d < aligned_k; _d += kAlign) {
            real_t const _XMMw = *(w + _d);


            // debug
//            real_t tmp=0.0;
//            real_t tmp1=0.0;
//            if((f1==0)|(pre_f1 != f1)){
//                tmp = _XMMw * _XMMv;
//            }else{
//                tmp1 = *( model.GetmidScore_Embedding(thread_i) + _e + _d);
//                tmp = tmp1 + _XMMw * _XMMv;
//            }
//            if(std::isnan(tmp) == 1){
//                std::cout<<"isnan:"<< std::endl;
//            }
//            if(std::isinf(tmp) == 1){
//                std::cout<<"isinf:"<< std::endl;
//            }
//            if(tmp >= 1.0e+10){
//                std::cout<<"large:"<< std::endl;
//            }
//            if(tmp <= -1.0e+10){
//                std::cout<<"small:"<< std::endl;
//            }


            if((f1==0)|(pre_f1 != f1)){
                *( model.GetmidScore_Embedding(thread_i) + _e + _d) = _XMMw * _XMMv;
            }else{
                *( model.GetmidScore_Embedding(thread_i) + _e + _d) += _XMMw * _XMMv;
            }

        }
        pre_f1 = f1;
    }


    /*********************************************************
     *
     *                                  fulllayer(laster):element_mean[aligned_k]
     *                                                /\
     *                                                ||
     *                                            -------------=w*x+b  the last fulllayer dont need active function
     *                                               ||
     *                                --------------------------------=active(w*x+b)
     *                                              ||
     *                   -------------------------------------------------------------=active(w*x+b)
     *                   ||                        ||                               ||
     *       filed1:element_mean[aligned_k]   filed2:element_mean[aligned_k]  filed3:element_mean[aligned_k]....
     *********************************************************/
    real_t* input=model.GetmidScore_OthersEmbedding(thread_i);
    index_t input_num=model.GetNum_midScore_OthersEmbedding(thread_i);
    for (index_t layer_j = 0; layer_j < num_fullLayer; ++layer_j) {
        real_t* output=model.GetmidScore_fulllayer(thread_i,layer_j);
        index_t output_num=model.GetNum_midScore_fulllayer(thread_i,layer_j);
        // w*x
        for(index_t in_i=0;in_i<input_num;++in_i){
            // get layer_j 's weights which is linked input_i
            real_t* w=model.GetFulllayer_w(layer_j,in_i);
            real_t in_v= *(input+in_i);
            for(index_t out_i=0; out_i<output_num; ++out_i){
                if(in_i==0){
                    *(output+out_i) =  *(w+out_i) * in_v;
                }else{
                    *(output+out_i) += *(w+out_i) * in_v;
                }
            }
        }
        // w*x+b
        real_t* b=model.GetFulllayer_b(layer_j);
        for(index_t out_i=0; out_i<output_num; ++out_i){
            *(output+out_i) += *(b+out_i);
        }
        // active(w*x+b)
        if(layer_j< (num_fullLayer-1)){
            for(index_t out_i=0; out_i<output_num; ++out_i){
                *(output+out_i) = relu(*(output+out_i));   //relu(w*x+b)
            }
        }
        // swap
        input     = output;
        input_num = output_num;
    }


    /*********************************************************
     *                                               out
     *                                               /\
     *                                               ||
     *                                  / ---------- * -----------\
     *                                 /                           \
     *                                /                             \
     *               filed0:element_mean[aligned_k]     fulllayer(laster):element_mean[aligned_k]
     *********************************************************/
     real_t t_all=0.0;
     for(index_t _d = 0; _d < aligned_k; _d += kAlign) {
         real_t tmp_targitrid = *(model.GetmidScore_TargitRidEmbedding(thread_i)+ _d);
         real_t tmp_fulllayer = *(model.GetmidScore_fulllayer(thread_i,num_fullLayer-1)+_d);
         t_all +=  tmp_targitrid * tmp_fulllayer;
     }

     // debug
//    if(std::isnan(t_all) == 1){
//        std::cout<<"isnan:"<< t_all << std::endl;
//    }

    return t_all;
}

// Calculate gradient and update current model parameters.
void YOUTUBEDNNScore::CalcGrad(
        const SparseRow* row,
        Model& model,
        index_t thread_i,
        real_t pg,
        real_t norm) {
  // Using sgd
  if (opt_type_.compare("sgd") == 0) {
    this->calc_grad_sgd(row, model, thread_i,pg, norm);
  }
  else {
    LOG(FATAL) << "Unknow optimization method: " << opt_type_;
  }
}

// Calculate gradient and update current model using sgd
// embedding is not trainable
void YOUTUBEDNNScore::calc_grad_sgd(
        const SparseRow* row,
        Model& model,
        index_t thread_i,
        real_t pg,
        real_t norm) {
  index_t aligned_k     = model.get_aligned_k();
  index_t num_fullLayer = model.GetNumFullLayerCell();


  /*********************************************************
   *                                               pg
   *                                               |
   *                                  / ---------- * -----------\
   *                                 /                           \
   *                                \/                           \/
   *               filed0:element_mean[aligned_k]     fulllayer(laster):element_mean[aligned_k]
   *********************************************************/
  /* *
  real_t g_tmp[aligned_k]={0.0};
  for(index_t _d = 0; _d < aligned_k; _d += kAlign) {
      // calc the g for filed0:element_mean[aligned_k]
      g_tmp[_d] = pg *  *(model.GetmidScore_fulllayer(thread_i,num_fullLayer-1)+ _d);
      // calc the g for fulllayer(laster):element_mean[aligned_k]
      *(model.GetmidScore_fulllayer(thread_i,num_fullLayer-1)+ _d) = pg *  *(model.GetmidScore_TargitRidEmbedding(thread_i)+ _d);
      *(model.GetmidScore_TargitRidEmbedding(thread_i)+ _d)        = g_tmp[_d];
  }
   * */
  for(index_t _d = 0; _d < aligned_k; _d += kAlign) {
      // calc the 局部梯度(激活函数前的梯度) g for fulllayer(laster):element_mean[aligned_k]
      *(model.GetmidScore_fulllayer(thread_i,num_fullLayer-1)+ _d) = pg *  *(model.GetmidScore_TargitRidEmbedding(thread_i)+ _d);
  }


  /*********************************************************
     *                                  fulllayer(laster):element_mean[aligned_k]
     *                                                ||
     *                                            -------------
     *                                               ||
     *                                --------------------------------
     *                                              ||
     *                   -------------------------------------------------------------
     *                   ||                        ||                               ||
     *                   \/                        \/                               \/
     *       filed1:element_mean[aligned_k]   filed2:element_mean[aligned_k]  filed3:element_mean[aligned_k]....
     *
     **********************************************************/
  index_t input_num = 0;
  real_t* input     = nullptr;
  for (int layer_j = num_fullLayer-1; layer_j >= 0; --layer_j) { // layer_j = 3,2,1,0
      if(layer_j!=0){
          input_num = model.GetNum_midScore_fulllayer(thread_i,layer_j-1);
          input     = model.GetmidScore_fulllayer(thread_i,layer_j-1);
      }else{
          input_num = model.GetNum_midScore_OthersEmbedding(thread_i);
          input     = model.GetmidScore_OthersEmbedding(thread_i);
      }
      index_t pass_g_num = model.GetNum_midScore_fulllayer(thread_i,layer_j);
      real_t* pass_g     = model.GetmidScore_fulllayer(thread_i,layer_j);   // 局部梯度(激活函数前的梯度)

      // update w and calc 局部梯度(激活函数前的梯度) g
      for(index_t in_i=0; in_i < input_num; in_i++){
          real_t* w   = model.GetFulllayer_w(layer_j,in_i);
          real_t in_v = *(input+in_i);
          // calc weight change
          real_t* w_change = model.GetFulllayer_w_change(thread_i,layer_j,in_i);
          for(index_t out_i=0; out_i < pass_g_num; out_i++){
              *(w_change+out_i) =learning_rate_ * in_v * *(pass_g+out_i);
          }
          // calc 局部梯度(激活函数前的梯度) g
          real_t g_sum_tmp=0.0;
          for(index_t out_i=0; out_i < pass_g_num; out_i++){
              g_sum_tmp += *(w+out_i) * *(pass_g+out_i);
          }
          // 把 局部梯度(激活函数前的梯度) g   , 使用 GetmidScore[即input] 来存储
          *(input+in_i) = ( in_v > 0.0) ? g_sum_tmp : 0.0;   // active function
          // update
          for(index_t out_i=0; out_i < pass_g_num; out_i++){
              *(w+out_i) -= *(w_change+out_i);
          }
      }

      // update b and calc 局部梯度(激活函数前的梯度) g
      real_t* b   = model.GetFulllayer_b(layer_j);
      real_t* b_change = model.GetFulllayer_b_change(thread_i,layer_j);
      // calc b change
      for(index_t out_i=0; out_i < pass_g_num; out_i++){
          *(b_change+out_i) = learning_rate_  * *(pass_g+out_i);
      }
      // update
      for(index_t out_i=0; out_i < pass_g_num; out_i++){
          *(b+out_i) -= *(b_change+out_i);
      }

  }
}

} // namespace xLearn
