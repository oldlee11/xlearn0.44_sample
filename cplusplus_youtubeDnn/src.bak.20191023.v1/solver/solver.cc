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
This file is the implementation of the Solver class.
*/

#include "src/solver/solver.h"

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <thread>

#include "src/base/stringprintf.h"
#include "src/base/split_string.h"
#include "src/base/timer.h"
#include "src/base/system.h"

namespace xLearn {

//------------------------------------------------------------------------------
//         _
//        | |
//   __  _| |     ___  __ _ _ __ _ __
//   \ \/ / |    / _ \/ _` | '__| '_ \
//    >  <| |___|  __/ (_| | |  | | | |
//   /_/\_\______\___|\__,_|_|  |_| |_|
//
//      xLearn   -- 0.44 Version --
//------------------------------------------------------------------------------
void Solver::print_logo() const {
  std::string logo = 
"----------------------------------------------------------------------------------------------\n"
                    "           _\n"
                    "          | |\n"
                    "     __  _| |     ___  __ _ _ __ _ __\n"
                    "     \\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
                    "      >  <| |___|  __/ (_| | |  | | | |\n"
                    "     /_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
                    "        xLearn   -- 0.44 Version --\n"
"----------------------------------------------------------------------------------------------\n"
"\n";
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << green << bold << logo << def << reset;
}

/******************************************************************************
 * Creater functions                                                          *
 ******************************************************************************/

// Create Reader by a given string
Reader* Solver::create_reader() {
  Reader* reader;
  std::string str = "memory";
  reader = CREATE_READER(str.c_str());
  if (reader == nullptr) {
    LOG(FATAL) << "Cannot create reader: " << str;
  }
  return reader;
}

// Create Score by a given string
Score* Solver::create_score() {
  Score* score;
  score = CREATE_SCORE(hyper_param_.score_func.c_str());
  if (score == nullptr) {
    LOG(FATAL) << "Cannot create score: " << hyper_param_.score_func;
  }
  return score;
}

// Create Loss by a given string
Loss* Solver::create_loss() {
  Loss* loss;
  loss = CREATE_LOSS(hyper_param_.loss_func.c_str());
  if (loss == nullptr) {
    LOG(FATAL) << "Cannot create loss: "
               << hyper_param_.loss_func;
  }
  return loss;
}

// Create Metric by a given string
Metric* Solver::create_metric() {
  Metric* metric;
  metric = CREATE_METRIC(hyper_param_.metric.c_str());
  // Note that here we do not cheack metric == nullptr
  // this is because we can set metric to "none", which 
  // means that we don't print any metric info.
  return metric;
}

/******************************************************************************
 * Functions for xlearn initialize                                            *
 ******************************************************************************/

// Initialize Solver
void Solver::Initialize(int argc, char* argv[]) {
  //  Print logo
  print_logo();
  // Check and parse command line arguments
  checker(argc, argv);
  // Initialize log file
  init_log();
  // Init train or predict
  init_train();
}


// Check and parse command line arguments
void Solver::checker(int argc, char* argv[]) {
  try {
    checker_.Initialize(hyper_param_.is_train, argc, argv);
    if (!checker_.check_cmd(hyper_param_)) {
      Color::print_error("Arguments error");
      exit(0);
    }
  } catch (std::invalid_argument &e) {
    printf("%s\n", e.what());
    exit(1);
  }
}


// Initialize log file
void Solver::init_log() {
  std::string prefix = get_log_file(hyper_param_.log_file);
  if (hyper_param_.is_train) {
    prefix += "_train";
  } else {
    prefix += "_predict";
  }
  InitializeLogger(StringPrintf("%s.INFO", prefix.c_str()),
              StringPrintf("%s.WARN", prefix.c_str()),
              StringPrintf("%s.ERROR", prefix.c_str()));
}

// Initialize training task
void Solver::init_train() {

  /*********************************************************
   *  Initialize thread pool                               *
   *********************************************************/
  size_t threadNumber = std::thread::hardware_concurrency();
  if (hyper_param_.thread_number != 0) {
    threadNumber = hyper_param_.thread_number;
  }
  pool_ = new ThreadPool(threadNumber);
  Color::print_info(
    StringPrintf("xLearn uses %i threads for training task.",threadNumber)
  );

  /*********************************************************
   *  Initialize Reader                                    *
   *********************************************************/
  Timer timer;
  timer.tic();
  Color::print_action("Read Problem ...");
  LOG(INFO) << "Start to init Reader";
  // Get the Reader list
  int num_reader = 0;
  std::vector<std::string> file_list;
  // do not use cross-validation
  num_reader += 1;  // training file
  CHECK_NE(hyper_param_.train_set_file.empty(), true);
  file_list.push_back(hyper_param_.train_set_file);
  if (!hyper_param_.validate_set_file.empty()) {
    num_reader += 1;  // validation file
    file_list.push_back(hyper_param_.validate_set_file);
  }
  LOG(INFO) << "Number of Reader: " << num_reader;
  reader_.resize(num_reader, nullptr);
  // Create Reader
  for (int i = 0; i < num_reader; ++i) {
    reader_[i] = create_reader();
    reader_[i]->SetBlockSize(hyper_param_.block_size);
    reader_[i]->SetSeed(hyper_param_.seed);
    reader_[i]->Initialize(file_list[i]);
    if (reader_[i] == nullptr) {
      Color::print_error(
              StringPrintf("Cannot open the file %s",file_list[i].c_str())
      );
      exit(0);
    }
    LOG(INFO) << "Init Reader: " << file_list[i];
   }

  /*********************************************************
   *  Read problem                                         *
   *********************************************************/
  DMatrix* matrix = nullptr;
  index_t max_feat = 0,max_field = 0;
  for (int i = 0; i < num_reader; ++i) {
    while(reader_[i]->Samples(matrix)) {
      int tmp = matrix->MaxFeat();
      if (tmp > max_feat) { max_feat = tmp; }
      tmp = matrix->MaxField();
      if (tmp > max_field) { max_field = tmp; }
    }
    // Return to the begining of target file.
    reader_[i]->Reset();
  }

  hyper_param_.num_feature = max_feat + 1;
  if (hyper_param_.num_feature == 0) {
    Color::print_error("Feature index is too large (overflow).");
    LOG(FATAL) << "Feature index is too large (overflow).";
  }
  LOG(INFO) << "Number of feature: " << hyper_param_.num_feature;
  Color::print_info(
    StringPrintf("Number of Feature: %d", hyper_param_.num_feature)
  );

  hyper_param_.num_field = max_field + 1;
  if (hyper_param_.num_field == 0) {
    Color::print_error("Filed index is too large (overflow).");
    LOG(FATAL) << "Filed index is too large (overflow).";
  }
  LOG(INFO) << "Number of field: " << hyper_param_.num_field;
  Color::print_info(
    StringPrintf("Number of Field: %d", hyper_param_.num_field)
  );


  Color::print_info(
    StringPrintf("Time cost for reading problem: %.2f (sec)",timer.toc())
  );

  /*********************************************************
   *  Initialize Model                                     *
   *********************************************************/
  timer.reset();
  timer.tic();
  Color::print_action("Initialize model ...");
  // Initialize parameters from reader
  if (hyper_param_.pre_model_file.empty()) {
    model_ = new Model();
    if (hyper_param_.opt_type.compare("sgd") == 0) {
      hyper_param_.auxiliary_size = 1;
    }
    model_->Initialize(hyper_param_.score_func,
                       hyper_param_.loss_func,
                       hyper_param_.num_feature,
                       hyper_param_.num_field,
                       hyper_param_.num_K,
                       hyper_param_.fullLayer_Cells,
                       hyper_param_.thread_number,
                       hyper_param_.auxiliary_size,
                       hyper_param_.model_scale);
  } else { // Initialize parameter from pre-trained model
    // bool para_isjustloadembedding=false; // re_trainning
    bool para_isjustloadembedding=true;// =true 表示仅仅初始化embedding  用于第一次训练
    model_ = new Model(hyper_param_.pre_model_file,hyper_param_,para_isjustloadembedding);
  }

  /*********************************************************
   *  Initialize score function                            *
   *********************************************************/
  score_ = create_score();
  score_->Initialize(hyper_param_.learning_rate,
                     hyper_param_.regu_lambda,
                     hyper_param_.alpha,
                     hyper_param_.beta,
                     hyper_param_.lambda_1,
                     hyper_param_.lambda_2,
                     hyper_param_.opt_type);
  LOG(INFO) << "Initialize score function.";
  /*********************************************************
   *  Initialize loss function                             *
   *********************************************************/
  loss_ = create_loss();
  loss_->Initialize(score_,
                    pool_,
                    hyper_param_.batch_size,
                    hyper_param_.norm,
                    hyper_param_.lock_free);
  loss_->Init_Atomic();
  LOG(INFO) << "Initialize loss function.";
  /*********************************************************
   *  Init metric                                          *
   *********************************************************/
  metric_ = create_metric();
  if (metric_ != nullptr) {
    metric_->Initialize(pool_);
  }
  LOG(INFO) << "Initialize evaluation metric.";
}


/******************************************************************************
 * Functions for xlearn start work                                            *
 ******************************************************************************/

// Start training or inference
void Solver::StartWork() {
  if (hyper_param_.is_train) {
    LOG(INFO) << "Start training work.";
    start_train_work();
  }
}

// Train
void Solver::start_train_work() {
  int epoch = hyper_param_.num_epoch;
  bool early_stop = hyper_param_.early_stop ;
  int stop_window = hyper_param_.stop_window;
  bool quiet = hyper_param_.quiet ;
  bool save_txt_model = true;
  if (hyper_param_.txt_model_file.compare("none") == 0 ) {
    save_txt_model = false;
  }
  Trainer trainer;
  trainer.Initialize(reader_,  /* Reader list */
                     epoch,
                     model_,
                     loss_,
                     metric_,
                     early_stop,
                     stop_window,
                     quiet);
  Color::print_action("Start to train ...");
/******************************************************************************
 * Original training without cross-validation                                 *
 ******************************************************************************/
  // The training process
  trainer.Train();
  // Save TXT model
  if (save_txt_model) {
    Timer timer;
    timer.tic();
    Color::print_action("Start to save txt model ...");
    trainer.SaveTxtModel(hyper_param_.txt_model_file);
    Color::print_info(
            StringPrintf("TXT Model file: %s", hyper_param_.txt_model_file.c_str())
    );
    Color::print_info(
            StringPrintf("Time cost for saving txt model: %.2f (sec)", timer.toc())
    );
  }
  Color::print_action("Finish training");

}


/******************************************************************************
 * Functions for xlearn finalization                                          *
 ******************************************************************************/

// Finalize xLearn
void Solver::Clear() {
  LOG(INFO) << "Clear the xLearn environment ...";
  Color::print_action("Clear the xLearn environment ...");
  // Clear model
  delete this->model_;
  // Clear Reader
  for (size_t i = 0; i < this->reader_.size(); ++i) {
    if (reader_[i] != nullptr) {
      delete reader_[i];
    }
  }
  reader_.clear();
}

} // namespace xLearn
