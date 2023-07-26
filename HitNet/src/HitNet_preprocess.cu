#include "HitNet_preprocess.h"
#include <opencv2/opencv.hpp>

__global__ void HitNet_preprocess_kernel(uint8_t* left,uint8_t* right,int width, int height, float* dst)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if ( position>= width*height) 
        return;

    int dx = position % width;
    int dy = position / width;

    float left_c0, left_c1, left_c2;
    float right_c0, right_c1, right_c2;

    uint8_t* v_left = left + (dy * width + dx) * 3;
    uint8_t* v_right= right + (dy * width + dx) * 3;

    left_c0 = (float)v_left[0]/255.0;
    left_c1 = (float)v_left[1]/255.0;
    left_c2 = (float)v_left[2]/255.0;

    right_c0 = (float)v_right[0]/255.0;
    right_c1 = (float)v_right[1]/255.0;
    right_c2 = (float)v_right[2]/255.0;


    //bgr to rgb 
    float left_t = left_c2;
    left_c2 = left_c0;
    left_c0 = left_t;

    float right_t = right_c2;
    right_c2 = right_c0;
    right_c0 = right_t;

    //rgbrgbrgb to rrrgggbbb
    int area = width * height;
    float* pdst_c0 = dst + dy * width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = left_c0;
    *pdst_c1 = left_c1;
    *pdst_c2 = left_c2;

    pdst_c0 = dst + dy * width + dx+3*width*height;
    pdst_c1 = pdst_c0 + area;
    pdst_c2 = pdst_c1 + area;
    *pdst_c0 = right_c0;
    *pdst_c1 = right_c1;
    *pdst_c2 = right_c2;

}

__global__ void HitNet_reprojectImageTo3D_kernel(uint8_t* left_img,float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=disparity_cols*disparity_rows)
    {
        return;
    }
    int col=tid%disparity_cols;
    int row=tid/disparity_cols;

    uint8_t* v = left_img + row * disparity_cols*3 + col * 3;
    float w=Q_device[14]*disparity[row*disparity_cols+col];

    for (size_t i = 0; i < 3; i++)
    {
        pointcloud[(row*disparity_cols+col)*6+i]=(Q_device[i*4]*col+Q_device[i*4+1]*row+Q_device[i*4+3]*1)/w;  
    }
    pointcloud[(row*disparity_cols+col)*6+3]=(float)v[2]; 
    pointcloud[(row*disparity_cols+col)*6+4]=(float)v[1];
    pointcloud[(row*disparity_cols+col)*6+5]=(float)v[0];
}



void HitNet_preprocess(uint8_t* left,uint8_t* right,int width, int height, float* dst, cudaStream_t stream) 
{

    int jobs = width * height;
    int threads = 256;
    int blocks = (jobs +threads-1)/threads;
    HitNet_preprocess_kernel<<<blocks, threads, 0, stream>>>(left,right,width,height,dst);
}

void HitNet_reprojectImageTo3D(uint8_t* left_img,float*disparity,float*pointcloud,float*Q_device,int disparity_rows,int disparity_cols)
{
    int jobs=disparity_rows*disparity_cols;
    int threads=256;
    int blocks=(jobs+threads-1)/threads;
    HitNet_reprojectImageTo3D_kernel<<<blocks,threads>>>(left_img,disparity,pointcloud,Q_device,disparity_rows,disparity_cols);
}