
/data/ming.li/workspace/TrainModel/api/fm/cplusplus_youtubeDnn_v2/cmake-build-debug/test/solver/train_main  \
    -tr /data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/v1/ctr_dateset.youtubeDnn.withoutuser.train.v1.txt \
    -te /data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/v1/ctr_dateset.youtubeDnn.withoutuser.test.v1.txt \
    -nthread 12 \
    -lr 0.01 \
    -epochs 5 \
    -batchsize 8 \
    -cells0 512 \
    -cells1 100 \
    -isjustloadembedding true \
    -premodel /data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn_withoutuser/init.modle.youtubeDnn.withoutuser.txt 
    


