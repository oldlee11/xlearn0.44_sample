//
// Created by ming.li on 19-12-3.
//

#ifndef YOUTUBEDNN_BASE_UTIL_H_
#define YOUTUBEDNN_BASE_UTIL_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>  // Linux, MacOSX, Cygwin and after VS2010 has this standard header.
#include <limits>

typedef long unsigned int size_t;

const int kAlign = 1;
const int kAlignByte = 4;




#ifdef _MSC_VER
typedef __int8  int8;
typedef __int16 int16;
typedef __int32 int32;
typedef __int64 int64;

typedef unsigned __int8  uint8;
typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef unsigned __int64 uint64;
#else
typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
#endif


//------------------------------------------------------------------------------
// We use 32-bits float to store the real number during computation,
// such as the modelParameter parameter and the gradient.
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32-bits unsigned integer to store the index
// of the feature and the modelParameter parameters.
//------------------------------------------------------------------------------
typedef uint32 index_t;

static const float kFloatMax = std::numeric_limits<float>::max();
static const float kFloatMin = std::numeric_limits<float>::min();


//------------------------------------------------------------------------------
// MetricInfo stores the evaluation metric information, which
// will be printed for users during the training.
//------------------------------------------------------------------------------
struct MetricInfo {
    real_t loss_val;    /* Loss value */
    real_t metric_val;  /* Metric value */
};

#endif //YOUTUBEDNN_UTIL_H
