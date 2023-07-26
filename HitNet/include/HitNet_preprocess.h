#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix 
{
  float value[6];
};



void HitNet_preprocess(uint8_t* left,uint8_t* right,int width, int height, float* dst, cudaStream_t stream);

void HitNet_reprojectImageTo3D(uint8_t* left_img,float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols);

#endif  // __PREPROCESS_H
