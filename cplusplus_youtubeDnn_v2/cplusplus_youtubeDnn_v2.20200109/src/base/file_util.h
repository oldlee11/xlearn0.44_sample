

/*
This file contains facilitlies to control the file.
*/

#ifndef YOUTUBEDNN_BASE_FILE_UTIL_H_
#define YOUTUBEDNN_BASE_FILE_UTIL_H_

#ifndef _MSC_VER
#include <unistd.h>
#else
#include "src/base/unistd.h"
#endif
#include <fcntl.h>
#include <string.h>
#include <stdio.h>

#include "src/base/util.h"



//------------------------------------------------------------------------------
// Useage:
//
//    std::string filename = "test_file";
//
//    /* (1) Check whether a file exists */
//    bool bo = FileExist(filename.c_str());
//
//    /* (2) Open file : 'r' for read, 'w' for write */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
//
//    /* (3) Close file */
//    Close(file_r);
//    Close(file_w);
//
//    /* (4) Get file size */
//    uint64 size_w = GetFileSize(file_w);
//    uint64 size_r = GetFileSize(file_r);
//
//    /* (5) Get one line from file */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    std::string str_line;
//    GetLine(file_r, str_line);
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Basic operations for a file
//------------------------------------------------------------------------------

// Max size of one line TXT data
static const uint32 kMaxLineSize = 10 * 1024 * 1024;  // 10 MB
// Max chunk size of hash block
static const uint32 kChunkSize = 1000 * 1024; // 1000 KB

// Check wether the file exists.
inline bool FileExist(const char *filename) {
  if (access(filename, F_OK) != -1) {
    return true;
  }
  std::cout<<"File: " << filename << " doesn't exists."<<"\n";
  return false;
}

// Open file using fopen() and return the file pointer.
// Args_mode : "w" for write and "r" for read
inline FILE *OpenFileOrDie(const char *filename, const char *mode) {
  FILE *input_stream = fopen(filename, mode);
  if (input_stream == nullptr) {
    std::cout<<"Cannot open file: " << filename << " with mode: " << mode<<"\n";
  }  
  return input_stream;
}

// Close file using fclose() by given the file pointer.
inline void Close(FILE *file) {
  if (fclose(file) == EOF) {
    std::cout<<"Error: invoke fclose()."<<"\n";
  }
}

// Return the size (byte) of a target file.
inline uint64 GetFileSize(FILE *file) {
  if (fseek(file, 0L, SEEK_END) != 0) {
    std::cout<<"Error: invoke fseek()."<<"\n";
  }
  // Note that we use uint64 here for big file
  uint64 total_size = ftell(file);
  if (total_size == -1) {
    std::cout<<"Error: invoke ftell()."<<"\n";
  }
  // Return to the head of file
  rewind(file);
  return total_size;
}

// Get one line of data from file by given a file pointer
inline void GetLine(FILE *file, std::string &str_line) {
  // char line[kMaxLineSize];       // 该方式只能对常量做固定大小的分配
  char* line=new char[kMaxLineSize];// kMaxLineSize不是常量 所以数组需要使用new来动态分配 并在最后delete
  fgets(line, kMaxLineSize, file);
  int read_len = strlen(line);
  if (line[read_len-1] != '\n') {
    std::cout<<"Encountered a too-long line: Cannot find the '\n' char. Please check the data."<<"\n";
  } else {
    line[read_len-1] = '\0';
    // Handle the format in DOS and Windows
    if (read_len > 1 && line[read_len-2] == '\r') {
      line[read_len-2] = '\0';
    }
  }
  str_line.assign(line);
  delete [] line;
}

// Read a block data from disk file to a buffer.
// Return the data size (byte) we read from the file.
// If we reach the end of the file, return 0.
inline size_t ReadDataFromDisk(FILE *file, char *buf, size_t len) {
    // Reach the end of the file
    if (feof(file)) {
        return 0;
    }
    size_t ret = fread(buf, 1, len, file);
    if (ret > len) {
      std::cout<<"Error: invoke fread()."<<"\n";
    }
    return ret;
}

#endif // XLEARN_BASE_FILE_UTIL_H_
