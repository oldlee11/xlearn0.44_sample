//
// Created by ming.li on 19-12-4.
//

#include "src/solver/solver.h"
#include "src/base/timer.h"

namespace youtubDnn {
    
    void Solver::Initialize(HyperParam& hyper_param){
        hyper_param_=hyper_param;
        // you can add checker for hyper_param code
        init_train();
    }

    /******************************************************************************
     * Initialize training task
     *****************************************************************************/
    void Solver::init_train() {
        /*********************************************************
         *  Initialize thread pool                               *
         *********************************************************/
        Timer timer;
        timer.tic();
        size_t threadNumber = std::thread::hardware_concurrency();
        if (hyper_param_.thread_number != 0) {
            threadNumber = hyper_param_.thread_number;
        }
        pool_ = new ThreadPool(threadNumber);
        std::cout<< "youtubeDnn uses "<< threadNumber <<"threads for training task." << "\n";

        /*********************************************************
         *  Initialize Reader                                    *
         *********************************************************/
        std::cout<< "Read Problem ..." <<"\n";
        std::cout<< "Start to init Reader" <<"\n";
        // Get the Train Reader
        if(hyper_param_.train_set_file.empty()) exit(0);
        train_reader_= new InmemReader();
        train_reader_->SetBlockSize(hyper_param_.block_size);
        train_reader_->SetSeed(hyper_param_.seed);
        train_reader_->Initialize(hyper_param_.train_set_file);
        if(train_reader_ == nullptr){
            std::cout<< "Cannot open the file "<< hyper_param_.train_set_file.c_str() <<"\n";
            exit(0);
        }
        std::cout<< "Init Reader: " << hyper_param_.train_set_file.c_str() <<"\n";
        // Get the Test Reader
        if(hyper_param_.test_set_file.empty()) exit(0);
        test_reader_= new InmemReader();
        test_reader_->SetBlockSize(hyper_param_.block_size);
        test_reader_->SetSeed(hyper_param_.seed);
        test_reader_->Initialize(hyper_param_.test_set_file);
        if(test_reader_ == nullptr){
            std::cout<< "Cannot open the file "<< hyper_param_.test_set_file.c_str() <<"\n";
            exit(0);
        }
        std::cout<< "Init Reader: " << hyper_param_.test_set_file.c_str() <<"\n";

        /*********************************************************
         *  Read problem                                         *
         *********************************************************/
        DMatrix* matrix = nullptr;
        index_t max_feat = 0,max_field = 0;
        while(train_reader_->Samples(matrix)) {
            int tmp = matrix->MaxFeat();
            if (tmp > max_feat) { max_feat = tmp; }
            tmp = matrix->MaxField();
            if (tmp > max_field) { max_field = tmp; }
        }
        // Return to the begining of target file.
        train_reader_->Reset();

        hyper_param_.num_feature = max_feat + 1;
        if (hyper_param_.num_feature == 0) {
            std::cout << "Feature index is too large (overflow)."<<"\n";
        }
        std::cout << "Number of feature:" << hyper_param_.num_feature <<"\n";

        hyper_param_.num_field = max_field + 1;
        if (hyper_param_.num_field == 0) {
            std::cout << "Filed index is too large (overflow)." <<"\n";
        }
        std::cout << "Number of field: " << hyper_param_.num_field <<"\n";
        
        std::cout << "Time cost for reading problem: " << timer.toc() <<" (sec)" << "\n";

        /*********************************************************
         *  Initialize Model                                     *
         *********************************************************/
        timer.reset();
        timer.tic();
        std::cout << "Initialize model ..." << "\n";
        // Initialize parameters from reader
        if (hyper_param_.pre_model_file.empty()) {
            model_ = new Model();
            model_->Initialize(hyper_param_.num_feature,
                               hyper_param_.num_field,
                               hyper_param_.num_K,
                               hyper_param_.fullLayer_Cells,
                               hyper_param_.thread_number,
                               hyper_param_.model_scale);
        } else { // Initialize parameter from pre-trained model
            model_ = new Model(hyper_param_.pre_model_file,hyper_param_);
        }

        /*********************************************************
         *  Initialize score function                            *
         *********************************************************/
        network_ = new Network();
        network_->Initialize(hyper_param_.learning_rate);
        std::cout << "Initialize network function." << "\n";
        
        /*********************************************************
         *  Initialize loss function                             *
         *********************************************************/
        loss_ =new CrossEntropyLoss();
        loss_->Initialize(network_,
                          pool_,
                          hyper_param_.batch_size);
        loss_->Init_Atomic();
        std::cout << "Initialize loss function." <<"\n";
        
        /*********************************************************
         *  Init metric                                          *
         *********************************************************/
        metric_ = new AccMetric();
        metric_->Initialize(pool_);
        std::cout << "Initialize evaluation metric." << "\n";
    }



    /******************************************************************************
     * Functions for start work                                            *
     ******************************************************************************/
    void Solver::StartWork() {
        std::cout << "Start training work." << "\n";
        bool save_txt_model = true;
        std::string txt_model_file=hyper_param_.train_set_file+".model";
        std::cout<< "Start to train ..." << "\n";
        // The training process
        Train();
        // Save TXT model
        Timer timer;
        timer.tic();
        std::cout<< "Start to save txt model " <<"\n";
        std::cout<<"Time cost for saving txt model: "<< timer.toc() <<" (sec)"<<"\n";
        std::cout<<"Finish training"<<"\n";
        // Save TXT model
        if (save_txt_model) {
            Timer timer;
            timer.tic();
            std::cout<<"Start to save txt model ..."<<"\n";
            SaveTxtModel(txt_model_file);
            std::cout<<"TXT Model file:"<< txt_model_file.c_str()<<"\n";
            std::cout<<"Time cost for saving txt model: "<<timer.toc() <<"(sec)"<<"\n";
        }
        std::cout<<"Finish training"<<"\n";

    }


    /*********************************************************
     *  Show train info                                      *
     *********************************************************/
    void Solver::show_average_metric(){
        std::cout<<"epoch"<<"(%)\t|\t"<<"tr_loss"<<"\t|\t"<<"te_loss"<<"\t|\t"<<"te_metric"<<"\t|\t"<<"time_cost"<<"\n";
    }
    void Solver::show_train_info(real_t tr_loss,
                                 real_t te_loss,
                                 real_t te_metric,
                                 real_t time_cost,
                                 index_t epoch,
                                 index_t epoch_
    ) {
        std::cout<<epoch<<"("<< static_cast<int>(epoch*1.0/epoch_*100) <<")\t|\t"<<tr_loss<<"\t|\t"<<te_loss<<"\t|\t"<<te_metric<<"\t|\t"<<time_cost<<"\n";
    }


    /*********************************************************
     *  Train for all ephos                                *
     *********************************************************/
    void Solver::Train() {
        MetricInfo te_info;
        show_average_metric();
        for (int n = 1; n <= hyper_param_.num_epoch; ++n) {
            Timer timer;
            timer.tic();
            // Calc grad and update model
            real_t tr_loss = calc_gradient();
            te_info = calc_metric();
            // show evaludation metric info
            show_train_info(tr_loss,
                            te_info.loss_val,
                            te_info.metric_val,
                            timer.toc(),
                            n,
                            hyper_param_.num_epoch);
            }
    }

    /*********************************************************
     *  Calc gradient and update model                       *
     *********************************************************/
    real_t Solver::calc_gradient() {
        loss_->Reset();
        train_reader_->Reset();
        DMatrix* matrix = nullptr;
        for (;;) {
            index_t tmp = train_reader_->Samples(matrix);
            if (tmp == 0) { break; }
            loss_->CalcGrad(matrix, *model_);
        }
        return loss_->GetLoss();
    }

    /*********************************************************
     *  Calc evaluation metric                               *
     *********************************************************/
    MetricInfo Solver::calc_metric() {
        DMatrix* matrix = nullptr;
        std::vector<real_t> pred;
        metric_->Reset();
        loss_->Reset();
        test_reader_->Reset();
        for (;;) {
            index_t tmp = test_reader_->Samples(matrix);
            if (tmp == 0) { break; }
            if (tmp != pred.size()) { pred.resize(tmp); }
            loss_->Predict(matrix, *model_, pred);
            loss_->Evalute(pred, matrix->Y);
            if (metric_ != nullptr) {
                metric_->Accumulate(matrix->Y, pred);
            }
        }
        //
        MetricInfo info;
        info.loss_val = loss_->GetLoss();
        if (metric_ != nullptr) {
            info.metric_val = metric_->GetMetric();
        }
        return info;
    }



    /******************************************************************************
     * Functions for xlearn finalization                                          *
     ******************************************************************************/
    // Finalize xLearn
    void Solver::Clear() {
        std::cout<< "Clear the xLearn environment ..." <<"\n";
        // Clear model
        delete this->model_;
        // Clear Reader
        delete train_reader_;
    }

}