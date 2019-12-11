//
// Created by ming.li on 19-12-3.
//

#ifndef YOUTUBEDNN_NETWORK_YOUTUBEDNN_NETWORK_H_
#define YOUTUBEDNN_NETWORK_YOUTUBEDNN_NETWORK_H_


#include <vector>

#include "src/modelParameter/parameters.h"
#include "src/base/util.h"
#include "src/reader/DMatrix.h"

namespace youtubDnn {
class Network {
public:
    // Constructor and Desstructor
    Network() { }
    ~Network() { }

    void Initialize(real_t learning_rate) {
        learning_rate_ = learning_rate;
    }

    // Given one exmaple and current modelParameter, this method
    // returns the Youtube DNN network.
    real_t CalcScore(const SparseRow* row,
                     Model& model,
                     index_t thread_i,
                     real_t norm = 1.0);

    // Calculate gradient
    // modelParameter parameters.
    void CalcGrad(const SparseRow* row,
                  Model& model,
                  index_t thread_i,
                  real_t pg,
                  real_t norm = 1.0);

    // update
    // void UpDate(Model& modelParameter,index_t thread_i);
    void UpDate_AllThreads(Model& model);



protected:
    // Calculate gradient and update modelParameter using sgd
    void calc_grad_sgd(const SparseRow* row,
                       Model& model,
                       index_t thread_i,
                       real_t pg,
                       real_t norm = 1.0);

    inline real_t relu(real_t input){
        return (input>0.0) ? input:0.0;
    }

    real_t learning_rate_;


};
}
#endif //YOUTUBEDNN_YOUTUBEDNN_NETWORK_H
