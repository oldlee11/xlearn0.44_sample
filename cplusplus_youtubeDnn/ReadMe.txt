

版本                   功能                         缺点
src.bak.20191017.v2    单线程可以争取训练           多线程不行
src.bak.20191018.v3     添加了batch_size 但是没有参数化(写死)
src.bak.20191023.v1    支持多进程 

网络
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
     
     /*********************************************************
     *                                               out
     *                                               /\
     *                                               ||
     *                                  / ---------- * -----------\
     *                                 /                           \
     *                                /                             \
     *               filed0:element_mean[aligned_k]     fulllayer(laster):element_mean[aligned_k]
     *********************************************************/
