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
This file defines the class of model parameters.
*/

#ifndef XLEARN_DATA_MODEL_PARAMETERS_H_
#define XLEARN_DATA_MODEL_PARAMETERS_H_

#include <string>

#include <math.h>

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/base/logging.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {


class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { free_model(); }

  // Initialize model from a checkpoint file.
  explicit Model(const std::string& filename,xLearn::HyperParam& hyper_param_);

  // Initialize model parameters to zero or using
  // a random distribution.
  void Initialize(const std::string& score_func,
              const std::string& loss_func,
              index_t num_feature,
              index_t num_field,
              index_t num_K,
              index_t *fullLayer_Cells,
              index_t threadNumber,
              index_t aux_size,
              real_t scale = 1.0);

  // Serialize model to a checkpoint file.
  void Serialize(const std::string& filename);

  // Serialize model to a TXT file.
  void SerializeToTXT(const std::string& filename);

  // Deserialize model from a checkpoint file.
  bool Deserialize(const std::string& filename);


  std::string getStringFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH);
  bool setStringFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH,std::string &var);
  bool setIndextFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH,index_t &var);
  bool setFloattFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH,float_t &var);
  index_t setFloattVectorFromString(std::ifstream &i_file,const std::string key, const size_t LINE_LENGTH,real_t* w);
  bool DeserializeFromTxt(const std::string& filename,xLearn::HyperParam& hyper_param_,bool isjustloadembedding);

  // Take a record of the best model during training.
  void SetBestModel();

  // Shrink back for getting the best model.
  void Shrink();

  // Get the pointer of latent factor.
  inline real_t* GetEmbedding_v(index_t feat_id,index_t aligned_k) { return embedding_v_ + feat_id * aligned_k; }
  inline real_t* GetFulllayer_w(index_t layer_j,index_t input_i,index_t aligned_k) { return fulllayer_w_.at(layer_j)+ input_i * aligned_k;}
  inline real_t* GetFulllayer_b(index_t layer_j) { return fulllayer_b_.at(layer_j);}
  /*
   * midScore_threads_[thread_i]  storge[连续内存存储]
   * |-----------------|--------------------------------------------------------------------------|---------------------------------------------------|
   * <filed0 embedding> <filed1 embedding mean> <filed2 embedding mean> .. <filedn embedding mean> <layer0 output> <layer1 output> ... <layer2 output>
   *  |          |        |                                                           /               |         |    |---------|
   *  |          |        |                                                          /                |---------|    ptr=GetmidScore_fulllayer(thread_i,1)
   *  |          |        |                                                         /                \/
   *  |          |        |--------------------------------------------------------/                 ptr=GetmidScore_fulllayer(thread_i,0)
   *  |          |        |
   *  |          |       \/
   *  |----------|       ptr=GetmidScore_OthersEmbedding(thread_i)
   *  |                  num=GetNum_midScore_OthersEmbedding(thread_i)
   * \/
   * ptr=GetmidScore_TargitRidEmbedding(thread_i)
   * num=GetNum_midScore_TargitRidEmbedding(thread_i)
   * */
  inline real_t* GetmidScore_Embedding(index_t thread_i) {
    return midScore_threads_.at(thread_i).at(0);
  }
  inline real_t* GetmidScore_TargitRidEmbedding(index_t thread_i) {
    return midScore_threads_.at(thread_i).at(0);
  }
  inline index_t GetNum_midScore_TargitRidEmbedding(index_t thread_i) {
    return get_aligned_k();
  }
  inline real_t* GetmidScore_OthersEmbedding(index_t thread_i) {
    return midScore_threads_.at(thread_i).at(1);
  }
  inline index_t GetNum_midScore_OthersEmbedding(index_t thread_i) {
    return get_aligned_k()*(num_field_-1);
  }
  inline real_t* GetmidScore_fulllayer(index_t thread_i,index_t layer_j) {
    return midScore_threads_.at(thread_i).at(2+layer_j);
  }
  inline index_t GetNum_midScore_fulllayer(index_t thread_i,index_t layer_j) {
    return midScore_num_threads_.at(thread_i).at(2+layer_j);
  }

  // Get the size of the latent factor.
  // For linear score this value equals zero.
  inline index_t GetNumEmbedding_v() { return embedding_num_v_; }
  inline std::vector<index_t> GetNumFulllayer_w() { return fulllayer_num_w_; }

  // Reset current model parameters.
  inline void Reset() { set_value(); }

  // Get score function type.
  inline std::string& GetScoreFunction() { return score_func_; }

  // Get the loss function type.
  inline std::string& GetLossFunction() { return loss_func_; }

  // Get the number of feature.
  inline index_t GetNumFeature() { return num_feat_; }

  // Get the number of field.
  inline index_t GetNumField() { return num_field_; }

  // Get the number of k.
  inline index_t GetNumK() { return num_K_; }

  // Get the number of each fullLayer
  inline index_t* GetFullLayerCell() { return fullLayer_Cells_ ;}
  // Get the layers in fulllayer
  inline index_t GetNumFullLayerCell(){ return num_fullLayer_Cells_ ;}

  // Get the aligned size of K.
  inline index_t get_aligned_k() {
    return (index_t)ceil((real_t)num_K_/kAlign)*kAlign;
  }

  // Get the aligned size of index_t.
  inline index_t get_aligned(index_t input){
      return (index_t)ceil((real_t)input/kAlign)*kAlign;
  }

 protected:
  /* Score function
  For now it can be 'linear', 'fm', or 'ffm' */
  std::string  score_func_;
  /* Loss function
  For now it can be 'squared' and 'cross-entropy' */
  std::string  loss_func_;

  index_t aux_size_;

  /* Number of feature
  Feature id is start from 0 */
  index_t  num_feat_;
  /* Number of field (Used in ffm)
  Field id is start from 0 */
  index_t  num_field_;
  /* Number of K (used in fm and ffm)
  Becasue we use SSE, so the real k should be aligned.
  User can get the aligned K by using get_aligned_k() */
  index_t  num_K_;

  /* Number cells for each fullLayer*/
  index_t *fullLayer_Cells_;
  index_t num_fullLayer_Cells_;


  index_t threadNumber_;

  /* Storing the parameter of embedding factor */
  real_t*  embedding_v_ = nullptr;
  index_t  embedding_num_v_;
  /* The following varibles are used for early-stopping */
  real_t*  embedding_best_v_ = nullptr;

  /* Storing the parameter of fulllayer weights */
  std::vector<real_t*> fulllayer_w_;
  std::vector<index_t> fulllayer_num_w_;
  std::vector<real_t*> fulllayer_b_;
  std::vector<index_t> fulllayer_num_b_;
  /* The following varibles are used for early-stopping */
  std::vector<real_t*> fulllayer_best_w_;
  std::vector<real_t*> fulllayer_best_b_;

  /* 存储 正向计算时的部分输出数据,用于反向计算 */
  std::vector<std::vector<real_t*>> midScore_threads_;
  std::vector<std::vector<index_t>> midScore_num_threads_;

  /* Used to init model parameters */
  real_t scale_;

  // Initialize the value of model parameters and gradient cache.
  void initial(bool set_value = false);

  // Reset the value of current model parameters.
  void set_value();

  // Serialize w, v, b to disk file.
  void serialize_w_v_b(FILE* file);

  // Deserialize w, v, b from disk file.
  void deserialize_w_v_b(FILE* file);

  // Free the allocated memory.
  void free_model();

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

}  // namespace xLearn

#endif  // XLEARN_DATA_MODEL_PARAMETERS_H_
