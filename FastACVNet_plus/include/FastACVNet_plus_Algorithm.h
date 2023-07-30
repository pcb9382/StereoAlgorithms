
#pragma once
#ifndef _FASTACVNET_PLUS_ALGORITHM_
#define _FASTACVNET_PLUS_ALGORITHM_
#ifdef _WINDOWS 
#define FASTACVNET_PLUS_ALGORITHM_EXPORTS
#ifdef FASTACVNET_PLUS_ALGORITHM_EXPORTS
#define FASTACVNET_PLUS_Algorithm_API __declspec(dllexport)
#else
#define FASTACVNET_PLUS_Algorithm_API __declspec(dllimport)
#endif
#else
#define FASTACVNET_PLUS_Algorithm_API
#endif
#endif 

#include<stdio.h>
#include<opencv2/opencv.hpp>
//Description   ModuleConfig           
//Params1                       
//Params2       			
//Return											
extern "C" FASTACVNET_PLUS_Algorithm_API void* Initialize(char* model_path,int gpu_id,char*calibration_path);

//Description   
//Params		img			
//Return        int          
extern "C" FASTACVNET_PLUS_Algorithm_API int RunFastACVNet_plus(void* p,cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity);

//Description   
//Params		img			
//Return        int          
extern "C" FASTACVNET_PLUS_Algorithm_API int RunFastACVNet_plus_RectifyImage(void* p,cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity);

//Description   
//Params		
//Return
extern "C" FASTACVNET_PLUS_Algorithm_API const char* Version(void* p);

//Description   
//Params		
//Return		int			
extern "C" FASTACVNET_PLUS_Algorithm_API int Release(void* p);


