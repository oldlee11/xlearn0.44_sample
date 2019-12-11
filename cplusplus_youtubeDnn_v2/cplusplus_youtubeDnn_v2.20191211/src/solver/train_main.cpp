//
// Created by ming.li on 19-12-4.
//


#include "src/solver/solver.h"
#include "src/modelParameter/hyperparameters.h"
int main(int argc, char* argv[]) {
    youtubDnn::HyperParam hyper_param;

    hyper_param.model_scale = 0.66;

    std::string arg_name;
    for (int i = 1; i < argc; i++){
        std::string argv_str(argv[i]);
        if((i%2==0)&(i>=2)){
            std::string arg_name(argv[i-1]);
            std::string arg_value_str(argv[i]);
            if(arg_name=="-tr"){
                hyper_param.train_set_file=arg_value_str; // "/data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/v1/ctr_dateset.youtubeDnn.withoutuser.train.v1.txt";
                std::cout<<"hyper_param.train_set_file:"<< hyper_param.train_set_file<<"\n";
            }else if(arg_name=="-te"){
                hyper_param.test_set_file=arg_value_str;  // "/data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/v1/ctr_dateset.youtubeDnn.withoutuser.test.v1.txt";
                std::cout<<"hyper_param.test_set_file:"<< hyper_param.test_set_file<<"\n";
            }else if(arg_name=="-nthread"){
                hyper_param.thread_number=std::stoi(arg_value_str);  // 12
                std::cout<<"hyper_param.thread_number:"<< hyper_param.thread_number<<"\n";
            }else if(arg_name=="-lr"){
                hyper_param.learning_rate=std::stof(arg_value_str);  // 0.01
                std::cout<<"hyper_param.learning_rate:"<< hyper_param.learning_rate<<"\n";
            }else if(arg_name=="-epochs"){
                hyper_param.num_epoch=std::stof(arg_value_str);      // 30
                std::cout<<"hyper_param.num_epoch:"<< hyper_param.num_epoch<<"\n";
            }else if(arg_name=="-batchsize"){
                hyper_param.batch_size=std::stof(arg_value_str);      // 8
                std::cout<<"hyper_param.batch_size:"<< hyper_param.batch_size<<"\n";
            }else if(arg_name=="-isjustloadembedding"){
                // true
                // false
                hyper_param.isjustloadembedding= (arg_value_str=="true");
                std::cout<<"hyper_param.isjustloadembedding:"<< hyper_param.isjustloadembedding<<"\n";
            }else if(arg_name=="-premodel"){
                // "/data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/init.modle.youtubeDnn.withoutuser.txt";
                // "/data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/v1/ctr_dateset.youtubeDnn.withoutuser.train.v1.txt.model";
                hyper_param.pre_model_file=arg_value_str;
                std::cout<<"hyper_param.pre_model_file:"<< hyper_param.pre_model_file<<"\n";
            }else if(arg_name=="-cells0"){
                hyper_param.fullLayer_Cells[0]=std::stoi(arg_value_str);  // 512
                std::cout<<"hyper_param.fullLayer_Cells[0]:"<< hyper_param.fullLayer_Cells[0]<<"\n";
            }else if(arg_name=="-cells1"){
                hyper_param.fullLayer_Cells[1]=std::stoi(arg_value_str);  // 100
                std::cout<<"hyper_param.fullLayer_Cells[1]:"<< hyper_param.fullLayer_Cells[1]<<"\n";
            }else{
                std::cout<<arg_name<<" is error"<<"\n";
            }
        }
    }

    youtubDnn::Solver sol=youtubDnn::Solver();
    sol.Initialize(hyper_param);
    sol.SetTrain();
    sol.StartWork();
    sol.Clear();

    return 0;


}
