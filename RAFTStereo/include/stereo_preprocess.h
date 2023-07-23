#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix 
{
  float value[6];
};



void RAFTStereo_preprocess(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream);

void cuda_reprojectImageTo3D(float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols);

#endif  // __PREPROCESS_H
