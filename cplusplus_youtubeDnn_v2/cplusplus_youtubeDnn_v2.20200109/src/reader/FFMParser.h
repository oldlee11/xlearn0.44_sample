//
// Created by ming.li on 19-12-2.
//

#ifndef YOUTUBEDNN_READER_PARSER_H_
#define YOUTUBEDNN_READER_PARSER_H_

#include <vector>
#include <string>
#include <cstring>
#include "src/base/util.h"
#include "src/reader/DMatrix.h"

namespace youtubDnn {
//------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
//------------------------------------------------------------------------------
class FFMParser {
    public:
        FFMParser() { }
        ~FFMParser() {  }

        // Wether this dataset contains label y ?
        inline void setLabel(bool label) {
            has_label_ = label;
        }

        // Set Splitor
        inline void setSplitor(const std::string& splitor) {
            splitor_ = splitor;
        }

        // The real parse function invoked by users.
        // If reset == true, Parser will invoke matrix.Reset();
        void Parse(char* buf,
                           uint64 size,
                           DMatrix& matrix,
                           bool reset = false) ;

    protected:
        // Get one line from memory buffer.
        uint64 get_line_from_buffer(char* line,
                                    char* buf,
                                    uint64 pos,
                                    uint64 size);

        /* True for training task and
        False for prediction task */
        bool has_label_;
        /* Split string for data items */
        std::string splitor_;
};
}

#endif //YOUTUBEDNN_PARSER_H
