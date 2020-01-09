//
// Created by ming.li on 19-12-4.
//

#ifndef YOUTUBEDNN_SOLVER_H
#define YOUTUBEDNN_SOLVER_H

#include <src/loss/metric.h>
#include "src/base/util.h"
#include "src/reader/InmemReader.h"
#include "src/network/network.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/modelParameter/hyperparameters.h"
#include "src/modelParameter/parameters.h"

namespace youtubDnn {
    class Solver {
    public:
        // Constructor and Destructor
        Solver()
                : network_(nullptr),
                  loss_(nullptr),
                  metric_(nullptr){ }
        ~Solver() { }

        // Initialize the xLearn environment through the
        // given hyper-parameters. This function will be
        // used for python API.
        void Initialize(HyperParam& hyper_param);

        // Ser train or predict
        void SetTrain() { hyper_param_.is_train = true; }

        // Start a training task or start an inference task.
        void StartWork();

        // Clear the xLearn environment.
        void Clear();

        // Save txt model to disk file
        void SaveTxtModel(const std::string& filename) {
                model_->SerializeToTXT(filename);
        }


    protected:
        /* Global hyper-parameters */
        HyperParam hyper_param_;
        /* Global model parameters */
        Model* model_;
        /* One Reader corresponds one data file */
        InmemReader* train_reader_;
        InmemReader* test_reader_;
        /* linear, fm or ffm ? */
        Network* network_;
        /* cross-entropy or squared ? */
        CrossEntropyLoss* loss_;
        /* acc, prec, recall, mae, etc */
        AccMetric* metric_;
        /* ThreadPool for multi-thread training */
        ThreadPool* pool_;
        /* predict results */
        std::vector<real_t> out_;

        // train function for all epochs
        void Train();

        void init_train();

        // Caculate gradient and update model.
        // Return training loss.
        real_t calc_gradient();

        // Calculate loss value and evaluation metric.
        MetricInfo calc_metric();

        // Calculate average metric for cross-validation
        void show_average_metric();

        // Print information during the training.
        void show_train_info(real_t tr_loss,
                             real_t te_loss,
                             real_t te_metric,
                             real_t time_cost,
                             index_t epoch,
                             index_t best_epoch,
                             index_t total_epochs);



    };
}

#endif //YOUTUBEDNN_SOLVER_H
