//
// Created by ming.li on 19-12-4.
//

#ifndef YOUTUBEDNN_MODELPARAMETER_YOUTUBEDNN_PARAMETERS_H_
#define YOUTUBEDNN_MODELPARAMETER_YOUTUBEDNN_PARAMETERS_H_

#include <string>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <cstring>

#include "src/base/util.h"
#include "src/modelParameter/hyperparameters.h"

namespace youtubDnn {
class Model{
public:
    Model() {};
    ~Model() {free_model();};
    void Initialize(index_t num_feature,
                    index_t num_field,
                    index_t num_K,
                    index_t *fullLayer_Cells,
                    index_t threadNumber,
                    real_t scale);

    // Initialize model from a checkpoint file.
    Model(const std::string& filename,youtubDnn::HyperParam& hyper_param_);

    // Take a record of the best model during training
    // current -> best
    void SetBestModel();

    // Shrink back for getting the best model
    // best -> current
    void Shrink();

    // Serialize model to a TXT file.
    void SerializeToTXT(const std::string& filename);
    std::string getStringFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH);
    bool setStringFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH,std::string &var);
    bool setIndextFromString(std::ifstream &i_file,const std::string key,const size_t LINE_LENGTH,index_t &var);
    index_t setFloattVectorFromString(std::ifstream &i_file,const std::string key, const size_t LINE_LENGTH,real_t* w);

    bool DeserializeFromTxt(const std::string& filename,youtubDnn::HyperParam& hyper_param_);
    bool DeserializeEmbeddingFromTxt(const std::string& filename);


    // get w and b
    inline real_t* GetEmbedding_v(index_t feat_id) { return embedding_v_ + feat_id * aligned_k_; }
    inline real_t* GetFulllayer_w(index_t layer_j,
                                  index_t input_i) { return fulllayer_w_.at(layer_j)+ input_i * (*(fullLayer_Cells_+layer_j));}
    inline real_t* GetFulllayer_w_change(index_t thread_i,
                                         index_t layer_j,
                                         index_t input_i) { return fulllayer_w_change_.at(thread_i).at(layer_j)+ input_i * (*(fullLayer_Cells_+layer_j));}
    inline real_t* GetFulllayer_b(index_t layer_j) { return fulllayer_b_.at(layer_j);}
    inline real_t* GetFulllayer_b_change(index_t thread_i,
                                         index_t layer_j) { return fulllayer_b_change_.at(thread_i).at(layer_j);}
    inline real_t* GetFulllayer_change_num(){return fulllayer_change_num_;}
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
    inline index_t GetNum_midScore_fulllayer(index_t layer_j) {
        return midScore_num_threads_.at(2+layer_j);
    }

    // Reset current model parameters.
    inline void Reset() { set_value(); }

    // Get the number of feature.
    inline index_t GetNumFeature() { return num_feat_; }

    // Get the layers in fulllayer
    inline index_t GetNumFullLayerCell(){ return num_fullLayer_Cells_ ;}

    // Get the aligned size of K.
    inline index_t get_aligned_k() {
        // return (index_t)ceil((real_t)num_K_/kAlign)*kAlign;
        return aligned_k_;
    }

    inline index_t GetthreadNumber(){
        return threadNumber_;
    }

    // Get the aligned size of index_t.
    inline index_t get_aligned(index_t input){
        return (index_t)ceil((real_t)input/kAlign)*kAlign;
    }


protected:

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
    index_t  aligned_k_;

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
    /* update w and b
     * fulllayer_w_  -= lr*fulllayer_w_change_
     * */
    std::vector<std::vector<real_t*>> fulllayer_w_change_;
    std::vector<std::vector<real_t*>> fulllayer_b_change_;
    real_t* fulllayer_change_num_;

    /* 存储 正向计算时的部分输出数据,用于反向计算 */
    std::vector<std::vector<real_t*>> midScore_threads_;
    std::vector<index_t> midScore_num_threads_;

    /* Used to init model parameters */
    real_t scale_;

    // Initialize the value of model parameters and gradient cache.
    void initial(bool set_value = false);

    // Reset the value of current model parameters.
    void set_value();

    // Free the allocated memory.
    void free_model();

};
}

#endif //YOUTUBEDNN_MODEL_PARAMETERS_H
