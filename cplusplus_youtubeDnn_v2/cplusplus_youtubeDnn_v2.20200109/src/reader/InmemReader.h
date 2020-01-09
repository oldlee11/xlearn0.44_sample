

#ifndef YOUTUBEDNN_READER_INMEMREADER_H_
#define YOUTUBEDNN_READER_INMEMREADER_H_

#include <string>
#include <vector>
#include <thread>
#include <algorithm> // for random_shuffle

#include "src/base/util.h"
#include "src/reader/DMatrix.h"
#include "src/reader/FFMParser.h"

namespace youtubDnn {

const size_t kDefautBlockSize = 500;  // 500 MB


// ------------------------------------------------------------------------------
// Sampling data from memory buffer.
// For in-memory smaplling, the Reader will automatically convert
// txt data to binary data, and uses this binary data in the next time.
//------------------------------------------------------------------------------
class InmemReader {
 public:
  // Constructor and Desstructor
  InmemReader() :
            shuffle_(false),
            bin_out_(true),
            block_size_(kDefautBlockSize),
            pos_(0) {
    }
  ~InmemReader() {  }

  // Pre-load all the data into memory buffer.
  void Initialize(const std::string& filename);

  // Sample data from the memory buffer.
  // Return the number of record in each samplling.
  // Samples() will return 0 when reaching end of the data.
  index_t Samples(DMatrix* &matrix);

  // Return to the begining of the data buffer.
  void Reset() { pos_ = 0; }

  // Free the memory of data matrix.
  void Clear() {
    data_buf_.Reset();
    data_samples_.Reset();
    if (block_ != nullptr) {
      delete [] block_;
    }
  }

  // Set the size of the block buffer.
  inline void SetBlockSize(size_t size) {
    block_size_ = size; 
  }

  // Set random see
  void SetSeed(int seed) {
    seed_ = seed;
  }

  // If shuffle data ?
  inline void SetShuffle(bool shuffle) {
    shuffle_ = shuffle;
    if (shuffle_ && !order_.empty()) {
      srand(seed_);
      random_shuffle(order_.begin(), order_.end());
    }
  }

  // Get data buffer
  inline DMatrix* GetMatrix() {
    return &data_buf_;
  }

 protected:
  /* Input file name */
  std::string filename_;
  /* Sample() returns this data sample */
  DMatrix data_samples_;
  /* Reader will load all the data into this buffer */
  DMatrix data_buf_;
  /* Parser for a block of memory buffer */
  FFMParser* parser_;
  /* If this data has label y ?
  This value will be set automitically in initialization */
  bool has_label_;
  /* If shuffle data ? */
  bool shuffle_;
  /* Generate bin file ? */
  bool bin_out_;
  /* Split string for data items */
  std::string splitor_;
  /* A block of memory to store the data */
  char* block_;
  /* Block size */
  size_t block_size_;
  /* Position for samplling */
  index_t pos_;
  /* Number of record at each samplling */
  index_t num_samples_;
  /* Random seed */
  int seed_ = 1;
  /* For random shuffle */
  std::vector<index_t> order_;


  // Check current file format and return
  // "libsvm", "ffm", or "csv".
  // Program crashes for unknow format.
  // This function will also check if current
  // data has the label y.
  std::string check_file_format();

  // Find the last '\n' in block and 
  // shrink back file pointer.
  void shrink_block(char* block, size_t* ret, FILE* file);

  // Initialize Reader from a new txt file.
  void init_from_txt();


};


} // namespace xLearn

#endif
