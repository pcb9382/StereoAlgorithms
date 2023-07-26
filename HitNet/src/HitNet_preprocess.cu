#include "HitNet_preprocess.h"
#include <opencv2/opencv.hpp>

__global__ void HitNet_preprocess_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) 
        return;

    int dx = position % src_width;
    int dy = position / src_width;

    float c0, c1, c2;

    uint8_t* v = src + dy * src_line_size + dx * 3;

    c0 = v[0];
    c1 = v[1];
    c2 = v[2];

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    // c0 = (c0 / 255.0f - 0.485) / 0.229;
    // c1 = (c1 / 255.0f - 0.456) / 0.224;
    // c2 = (c2 / 255.0f - 0.406) / 0.225;

    //rgbrgbrgb to rrrgggbbb
    int area = src_width * src_height;
    float* pdst_c0 = dst + dy * src_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

__global__ void cuda_reprojectImageTo3D_kernel(float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=disparity_cols*disparity_rows)
    {
        return;
    }
    int col=tid%disparity_cols;
    int row=tid/disparity_cols;
    float w=Q_device[14]*disparity[row*disparity_cols+col];
    if (w>0)
    {
        for (size_t i = 0; i < 3; i++)
        {
            pointcloud[(row*disparity_cols+col)*3+i]=(Q_device[i*4]*col+Q_device[i*4+1]*row+Q_device[i*4+3]*1)/w;  
        }
    }
}



void HitNet_preprocess(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream) 
{

    int jobs = src_width * src_height;
    int threads = 256;
    int blocks = (jobs +threads-1)/threads;
    HitNet_preprocess_kernel<<<blocks, threads, 0, stream>>>(src, src_width*3, src_width, src_height, dst, jobs);
}

void cuda_reprojectImageTo3D(float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols)
{
    int jobs=disparity_rows*disparity_cols;
    int threads=256;
    int blocks=(jobs+threads-1)/threads;
    cuda_reprojectImageTo3D_kernel<<<blocks,threads>>>(disparity,pointcloud,Q_device,disparity_rows,disparity_cols);
}