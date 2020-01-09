//
// Created by ming.li on 19-12-4.
//

#ifndef YOUTUBEDNN_CROSS_ENTROPY_LOSS_H
#define YOUTUBEDNN_CROSS_ENTROPY_LOSS_H

#include <vector>
#include <string>
#include <thread>
#include <atomic>

#include "src/base/util.h"
#include "src/base/thread_pool.h"
#include "src/network/network.h"
#include "src/modelParameter/parameters.h"

namespace youtubDnn {
    class CrossEntropyLoss {
    public:
        // Constructor and Desstructor
        CrossEntropyLoss() : loss_sum_(0), total_example_(0) { };
        ~CrossEntropyLoss() { }
        // This function needs to be invoked before using this class
        void Initialize(Network* network,
                        ThreadPool* pool,
                        index_t batch_size) {
            network_ = network;
            pool_ = pool;
            threadNumber_ = pool_->ThreadNumber();
            batch_size_ = batch_size;
        }

        void Init_Atomic();

        // Given predictions and labels, accumulate loss value.
        void Evalute(const std::vector<real_t>& pred,
                             const std::vector<real_t>& label);

        // Given data sample and current model, return predictions.
        void Predict(const DMatrix* data_matrix,
                     Model& model,
                             std::vector<real_t>& pred);

        // Given data sample and current model, calculate gradient
        // and update current model parameters.
        // This function will also acummulate loss value.
        void CalcGrad(const DMatrix* data_matrix,Model& model);

        // Return the calculated loss value
        real_t GetLoss() {
            return loss_sum_ / total_example_;
        }

        // Reset loss_sum_ and total_example_
        void Reset() {
            loss_sum_ = 0;
            total_example_ = 0;
        }


    protected:
        /* The score function, including LinearScore,
        FMScore, FFMScore, etc */
        Network* network_;
        /* Thread pool for multi-thread training */
        ThreadPool* pool_;
        /* Number of thread in thread pool */
        size_t threadNumber_;

        /* Used to store the accumulate loss */
        real_t loss_sum_;
        /* Used to store the number of example */
        index_t total_example_;

        /* Mini-batch size */
        index_t batch_size_;

        /*
         * 每个线程可以运行的逻辑分为bp(计算梯度变化)和update(把梯度变化更新到参数中)
         * 其中update必须只能有一个线程来运行,且在运行update过程中不能有线程运行bp
         * 运行update的线程的顺序是[0,threadNumber_-1] 依次循环运行
        */
        // 现在是否在运行update程序？
        std::atomic<bool> isUpDateing_;
        // 那个线程正在或即将运行update程序？ 保证只能有1个thread线程运行update
        std::atomic<index_t> update_lock_;
        // 每个线程是否在执行bp？
        std::vector<std::atomic<bool>*> isBPing_;
        // 原子操作 定义 有的扯 不能直接在vector中
        std::atomic<bool> isBPing_Thread_0_;
        std::atomic<bool> isBPing_Thread_1_;
        std::atomic<bool> isBPing_Thread_2_;
        std::atomic<bool> isBPing_Thread_3_;
        std::atomic<bool> isBPing_Thread_4_;
        std::atomic<bool> isBPing_Thread_5_;
        std::atomic<bool> isBPing_Thread_6_;
        std::atomic<bool> isBPing_Thread_7_;
        std::atomic<bool> isBPing_Thread_8_;
        std::atomic<bool> isBPing_Thread_9_;
        std::atomic<bool> isBPing_Thread_10_;
        std::atomic<bool> isBPing_Thread_11_;
        std::atomic<bool> isBPing_Thread_12_;
        std::atomic<bool> isBPing_Thread_13_;
        std::atomic<bool> isBPing_Thread_14_;
        std::atomic<bool> isBPing_Thread_15_;

    };
}
#endif //YOUTUBEDNN_CROSS_ENTROPY_LOSS_H
