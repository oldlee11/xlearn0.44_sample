//
// Created by ming.li on 19-12-4.
//

#ifndef YOUTUBEDNN_YOUTUBEDNN_HYPERPARAMETERS_H_
#define YOUTUBEDNN_YOUTUBEDNN_HYPERPARAMETERS_H_

#include "src/base/util.h"
#include "src/reader/DMatrix.h"
#include <string>

namespace youtubDnn {
    struct HyperParam {
        //------------------------------------------------------------------------------
// Baisc parameters for current task
//------------------------------------------------------------------------------
        /* Train or Predict.
        True for train, and false for predict. */
        bool is_train = true;
        /* Number of thread existing in the thread pool */
        int thread_number = 1;
//------------------------------------------------------------------------------
// Parameters for optimization method
//------------------------------------------------------------------------------
        /* Hyper param for init model parameters */
        real_t model_scale = 0.66;
        /* Learning rate */
        real_t learning_rate = 0.2;
        /* Number of epoch.
        This value could be changed in early-stop */
        index_t num_epoch = 10;

//------------------------------------------------------------------------------
// Parameters for dataset
//------------------------------------------------------------------------------
        /* Number of feature */
        index_t num_feature = 0;
        /* Number of total model parameters */
        index_t num_param = 0;
        /* Number of lateny factor for fm and ffm */
        index_t num_K = 4;
        /* Number of field, used by ffm tasks */
        index_t num_field = 0;
        /* Nuber cells for each fullLayer */
        index_t fullLayer_Cells[2]={0,0};
        index_t batch_size=8;
        /* Filename of training dataset
        We must set this value in training task. */
        std::string train_set_file;
        /* Filename of test dataset
        We must set this value in predication task. */
        std::string test_set_file;
        /* DMatrix pointer for train*/
        youtubDnn::DMatrix* train_dataset = nullptr;
        /* DMatrix pointer for test*/
        youtubDnn::DMatrix* test_dataset = nullptr;

        /* Filename of model checkpoint
        On default, model_file = train_set_file + ".model" */
        std::string model_file;
        /* Pre-trained model for online learning */
        std::string pre_model_file;
        /* Init embedding */
        std::string init_embedding_file;
        /* Filename of log file */
#ifndef _MSC_VER
        std::string log_file = "/tmp/youtubeDnn_log";
#else
        std::string log_file = "youtubeDnn_log";
#endif
        /* Block size for on-disk training */
        int block_size = 500;  // 500 MB
        /* Random seed to shuffle data set */
        int seed = 1;
        /* If generate prediction file */
        //bool res_out = true;
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
        /* Convert predition output to 0 and 1 */
        // bool sign = false;
        /* Convert predition output using sigmoid */
        // bool sigmoid = false;
        /* True for using early-stop and False for not */
        bool early_stop = true;
        /* Early stop window size */
        int stop_window = 2;

    };
}

#endif //YOUTUBEDNN_YOUTUBEDNN_HYPERPARAMETERS_H
