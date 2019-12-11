//
// Created by ming.li on 19-11-29.
//

#include "src/reader/InmemReader.h"
int main(int argc, char* argv[]) {
    youtubDnn::InmemReader reader_obj = youtubDnn::InmemReader();
    reader_obj.Initialize("/data/ming.li/workspace/TrainModel/api/fm/demo/cplusplus_youtubeDnn/classification/kuwo/v1.sample/ctr_dateset.youtubeDnn.test.sample.v1.txt");
    reader_obj.SetShuffle(true);
    reader_obj.Reset();
    youtubDnn::DMatrix* matrix = nullptr;
    for (;;) {
        index_t tmp = reader_obj.Samples(matrix);
        if (tmp == 0) { break; }
    }
    return 0;
}

