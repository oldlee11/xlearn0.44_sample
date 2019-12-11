
/*
This file is the implementation of Reader class.
*/

#include "InmemReader.h"
#include "src/base/file_util.h"
#include "src/base/split_string.h"
#include "FFMParser.h"

#include <string.h>
#include <algorithm> // for random_shuffle


namespace youtubDnn {

// Check current file format and
// return 'libsvm', 'libffm', or 'csv'.
// This function will also check if current
// data has the label y.
std::string InmemReader::check_file_format() {
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
    std::cout<< "File format error!"<<"\n";
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
  exit(0);
}

// Find the last '\n' in block, and shrink back file pointer
void InmemReader::shrink_block(char* block, size_t* ret, FILE* file) {
  // Find the last '\n'
  size_t index = *ret-1;
  while (block[index] != '\n') { index--; }
  // Shrink back file pointer
  fseek(file, index-*ret+1, SEEK_CUR);
  // The real size of block
  *ret = index + 1;
}

// Pre-load all the data into memory buffer (data_buf_).
// Note that this funtion will first check whether we 
// can use the existing binary file. If not, reader will 
// generate one automatically.
void InmemReader::Initialize(const std::string& filename) {
  filename_ = filename;
  // Allocate memory for block
  try {
    this->block_ = (char*)malloc(block_size_*1024*1024);
  } catch (std::bad_alloc&) {
    std::cout<< "Cannot allocate enough memory for data block. Block size: "
                   << block_size_ << "MB. "
                   << "You set change the block size via configuration."<<"\n";
  }
  init_from_txt();
}

// Pre-load all the data to memory buffer from txt file.
void InmemReader::init_from_txt() {
  // Init parser_
  std::string fileformat=check_file_format().c_str();
  if(fileformat=="libffm"){
      parser_ = new FFMParser();
  }else{
      std::cout<<"check_file_format is error"<<"\n";
      exit(0);
  }
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




}  // namespace xLearn
