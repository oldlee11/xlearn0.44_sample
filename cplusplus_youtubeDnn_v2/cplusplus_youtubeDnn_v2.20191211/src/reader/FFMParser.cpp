//
// Created by ming.li on 19-12-2.
//

#include "src/reader/FFMParser.h"
#include "src/base/split_string.h"
#include "src/base/util.h"
#include <stdlib.h>

namespace youtubDnn {
    // Max size of one line TXT data
    static const uint32 kMaxLineSize = 10 * 1024 * 1024;  // 10 MB

    static char line_buf[kMaxLineSize];

    // Get one line from memory buffer
    uint64 FFMParser::get_line_from_buffer(char* line,
                                        char* buf,
                                        uint64 pos,
                                        uint64 size) {
        if (pos >= size) { return 0; }
        uint64 end_pos = pos;
        while (end_pos < size && buf[end_pos] != '\n') { end_pos++; }
        uint64 read_size = end_pos - pos + 1;
        if (read_size > kMaxLineSize) {
            std::cout<<"Encountered a too-long line.  Please check the data."<<"\n";
        }
        memcpy(line, buf+pos, read_size);
        line[read_size - 1] = '\0';
        if (read_size > 1 && line[read_size - 2] == '\r') {
            // Handle some txt format in windows or DOS.
            line[read_size - 2] = '\0';
        }
        return read_size;
    }

    //------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
// idx can start from 0
//------------------------------------------------------------------------------
    void FFMParser::Parse(char* buf,
                          uint64 size,
                          DMatrix& matrix,
                          bool reset) {
        // Clear the data matrix
        if (reset) {
            matrix.Reset();
        }
        // Parse every line
        uint64 pos = 0;
        for (;;) {
            uint64 rd_size = get_line_from_buffer(line_buf, buf, pos, size);
            if (rd_size == 0) break;
            pos += rd_size;
            matrix.AddRow();
            int i = matrix.row_length - 1;
            // Add Y
            if (has_label_) {  // for training task
                char *y_char = strtok(line_buf, splitor_.c_str());
                matrix.Y[i] = atof(y_char);
            } else {  // for predict task
                matrix.Y[i] = -2;
            }
            // Add features
            real_t norm = 0.0;
            // The first element
            if (!has_label_) {
                char *field_char = strtok(line_buf, ":");
                char *idx_char = strtok(nullptr, ":");
                char *value_char = strtok(nullptr, splitor_.c_str());
                if (idx_char != nullptr && *idx_char != '\n') {
                    index_t idx = atoi(idx_char);
                    real_t value = atof(value_char);
                    index_t field_id = atoi(field_char);
                    matrix.AddNode(i, idx, value, field_id);
                    norm += value*value;
                }
            }
            // The remain elements
            for (;;) {
                char *field_char = strtok(nullptr, ":");
                char *idx_char = strtok(nullptr, ":");
                char *value_char = strtok(nullptr, splitor_.c_str());
                if (field_char == nullptr || *field_char == '\n') {
                    break;
                }
                index_t idx = atoi(idx_char);
                real_t value = atof(value_char);
                index_t field_id = atoi(field_char);
                matrix.AddNode(i, idx, value, field_id);
                norm += value*value;
            }
            norm = 1.0f / norm;
            matrix.norm[i] = norm;
        }
    }

}