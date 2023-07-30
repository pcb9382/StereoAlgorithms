#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "FastACVNet_plus_preprocess.h"

#define CRESTEREO_BATCH_SIZE 1

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

using namespace nvinfer1;

struct  CalibrationParam
{
	cv::Mat intrinsic_left;
	cv::Mat distCoeffs_left;
	cv::Mat	intrinsic_right;
	cv::Mat distCoeffs_right;
	cv::Mat R;
	cv::Mat T;
	cv::Mat R_L;
	cv::Mat R_R;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;
};
class FastACVNet_plus
{
private:
    /* data */
public:
    FastACVNet_plus();
    ~FastACVNet_plus();
    int Initialize(std::string model_path,int gpu_id,CalibrationParam&Calibrationparam);
    int RunFastACVNet_plus(cv::Mat&rectifyImageL2,cv::Mat&rectifyImageR2,float*pointcloud,cv::Mat&DisparityMap);
    int Release();


public:
    // stuff we know about the network and the input/output blobs
    int INPUT_H;
    int INPUT_W;
    int OUTPUT_SIZE;
    char* INPUT_BLOB_NAME1;
    char* INPUT_BLOB_NAME2 ;
    char* OUTPUT_BLOB_NAME;
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine ; 
    IExecutionContext* context ;
    float* buffers[3];

    int inputIndex1;
    int inputIndex2;
    int outputIndex;
    float *flow_up;
    float * disparity_data;
    // Create stream
    cudaStream_t stream;
    uint8_t* img_left_host = nullptr;
    uint8_t* img_right_host = nullptr;
    uint8_t* img_left_device = nullptr;
    uint8_t* img_right_device = nullptr;
    float*PointCloud_devide=nullptr;
    float*Q_device;
    float *Calibrationparam_Q;
    CalibrationParam Calibrationparam;

public:
    inline bool file_exists (const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
};
