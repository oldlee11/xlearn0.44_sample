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
This file is the implementation of Reader class.
*/

#include "src/reader/reader.h"

#include <string.h>
#include <algorithm> // for random_shuffle

#include "src/base/file_util.h"
#include "src/base/split_string.h"
#include "src/base/format_print.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_reader_registry, Reader);
REGISTER_READER("memory", InmemReader);


// Check current file format and
// return 'libsvm', 'libffm', or 'csv'.
// This function will also check if current
// data has the label y.
std::string Reader::check_file_format() {
#ifndef _MSC_VER
  FILE* file = OpenFileOrDie(filename_.c_str(), "r");
#else
  FILE* file = OpenFileOrDie(filename_.c_str(), "rb");
#endif
  // get the first line of data
  std::string data_line;
  GetLine(file, data_line);
  Close(file);
  // Find the split string
  int space_count = 0;
  int table_count = 0;
  int comma_count = 0;
  for (size_t i = 0; i < data_line.size(); ++i) {
    if (data_line[i] == ' ') {
      space_count++;
    } else if (data_line[i] == '\t') {
      table_count++;
    } else if (data_line[i] == ',') {
      comma_count++;
    }
  }
  if (space_count > table_count && 
      space_count > comma_count) {
    splitor_ = " ";
  } else if (table_count > space_count &&
             table_count > comma_count) {
    splitor_ = "\t";
  } else if (comma_count > space_count &&
             comma_count > table_count) {
    splitor_ = ",";
  } else {
    LOG(FATAL) << "File format error!";
  }
  // Split the first line of data
  std::vector<std::string> str_list;
  SplitStringUsing(data_line, splitor_.c_str(), &str_list);
  // has y?
  size_t found = str_list[0].find(":");
  if (found != std::string::npos) {  // find ":", no label
    has_label_ = false;
  } else {
    has_label_ = true;
  }
  // check file format
  int count = 0;
  for (int i = 0; i < str_list[1].size(); ++i) {
    if (str_list[1][i] == ':') {
      count++;
    }
  }
  if (count == 1) {
    return "libsvm";
  } else if (count == 2) {
    return "libffm";
  } else if (count == 0){
    return "csv";
  }
  Color::print_error("Unknow file format");
  exit(0);
}

// Find the last '\n' in block, and shrink back file pointer
void Reader::shrink_block(char* block, size_t* ret, FILE* file) {
  // Find the last '\n'
  size_t index = *ret-1;
  while (block[index] != '\n') { index--; }
  // Shrink back file pointer
  fseek(file, index-*ret+1, SEEK_CUR);
  // The real size of block
  *ret = index + 1;
}

//------------------------------------------------------------------------------
// Implementation of InmemReader
//------------------------------------------------------------------------------

// Pre-load all the data into memory buffer (data_buf_).
// Note that this funtion will first check whether we 
// can use the existing binary file. If not, reader will 
// generate one automatically.
void InmemReader::Initialize(const std::string& filename) {
  CHECK_NE(filename.empty(), true)
  filename_ = filename;
  Color::print_info("First check if the text file has been already converted to binary format.");


  Color::print_info(
              StringPrintf("Binary file (%s.bin) NOT found. Convert text file to binary file.",filename_.c_str())
  );
  // Allocate memory for block
  try {
    this->block_ = (char*)malloc(block_size_*1024*1024);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for data block. Block size: "
                   << block_size_ << "MB. "
                   << "You set change the block size via configuration.";
  }
  init_from_txt();
}

// Pre-load all the data to memory buffer from txt file.
void InmemReader::init_from_txt() {
  // Init parser_                       
  parser_ = CreateParser(check_file_format().c_str());
  if (has_label_) parser_->setLabel(true);
  else parser_->setLabel(false);
  // Set splitor
  parser_->setSplitor(this->splitor_);
  // Convert MB to Byte
  uint64 read_byte = block_size_ * 1024 * 1024;
  // Open file
#ifndef _MSC_VER
  FILE* file = OpenFileOrDie(filename_.c_str(), "r");
#else
  FILE* file = OpenFileOrDie(filename_.c_str(), "rb");
#endif
  // Read until the end of file
  for (;;) {
    // Read a block of data from disk file
    size_t ret = ReadDataFromDisk(file, block_, read_byte);
    if (ret == 0) {
      break;
    } else if (ret == read_byte) {
      // Find the last '\n', and shrink back file pointer
      this->shrink_block(block_, &ret, file);
    } // else ret < read_byte: we don't need shrink_block()
    parser_->Parse(block_, ret, data_buf_, false);
  }
  data_buf_.SetHash(HashFile(filename_, true),HashFile(filename_, false));
  data_buf_.has_label = has_label_;
  // Init data_samples_ 
  num_samples_ = data_buf_.row_length;
  data_samples_.ReAlloc(num_samples_, has_label_);
  // for shuffle
  order_.resize(num_samples_);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
  delete [] block_;
  Close(file);
}

// Smaple data from memory buffer.
index_t InmemReader::Samples(DMatrix* &matrix) {
  for (int i = 0; i < num_samples_; ++i) {
    if (pos_ >= data_buf_.row_length) {
      // End of the data buffer
      if (i == 0) {
        if (shuffle_) {
          srand(this->seed_+1);
          random_shuffle(order_.begin(), order_.end());
        }
        matrix = nullptr;
        return 0;
      }
      break;
    }
    // Copy data between different DMatrix.
    data_samples_.row[i] = data_buf_.row[order_[pos_]];
    data_samples_.Y[i] = data_buf_.Y[order_[pos_]];
    data_samples_.norm[i] = data_buf_.norm[order_[pos_]];
    pos_++;
  }
  matrix = &data_samples_;
  return num_samples_;
}

// Return to the begining of the data buffer.
void InmemReader::Reset() { pos_ = 0; }


}  // namespace xLearn
