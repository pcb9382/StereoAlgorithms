#ifndef __FASTACVNET_plus_PREPROCESS_H
#define __FASTACVNET_PLUS_PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix 
{
  float value[6];
};



void FastACVNet_plus_preprocess(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream);

void FastACVNet_plus_reprojectImageTo3D(uint8_t* left_img,float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols);

#endif  // __CRESTEREO_PREPROCESS_H
