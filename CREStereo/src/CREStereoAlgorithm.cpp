#include<iostream>
#include<opencv2/opencv.hpp>
#include"CREStereoAlgorithm.h"
#include"TRTCREStereo.h"

class CREStereoAlgorithm
{
private:
    CREStereo crestereo;
public:
    CREStereoAlgorithm(/* args */);
    ~CREStereoAlgorithm();
   				
    int Initialize(char* model_path,int gpu_id,char*calibration_path);
    int RunCREStereo(cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity);
    int RunCREStereo_RectifyImage(cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity);
    int Release();
    void ReadObjectYml(const char* filename, CalibrationParam&Calibrationparam);
    int RectifyImage(cv::Mat&Image_src,cv::Mat&rectifyImageL2,cv::Mat&rectifyImageR2);
    int RectifyImage(cv::Mat&rectifyImageL2,cv::Mat&rectifyImageR2);
public:
    CalibrationParam Calibrationparam;
    int initflag=-1;
};

CREStereoAlgorithm::CREStereoAlgorithm(/* args */)
{
}

CREStereoAlgorithm::~CREStereoAlgorithm()
{
}


int CREStereoAlgorithm::Initialize(char* model_path,int gpu_id,char*calibration_path)
{
   std::string model_path_str=model_path;
   if(crestereo.file_exists(calibration_path))
   {
        ReadObjectYml(calibration_path,Calibrationparam);
   }
   else
   {
        std::cout<<"calibration_path does not exist!!!"<<std::endl;
   }
   initflag=crestereo.Initialize(model_path_str,gpu_id,Calibrationparam);
   if (initflag!=0)
   {
        std::cout<<"init failed!!"<<std::endl;
        return -1;
   }
   initflag=0;
   std::cout<<"init successed!"<<std::endl;
   return 0;
  
}


int CREStereoAlgorithm::RunCREStereo_RectifyImage(cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity)
{
    if (initflag!=0)
    {
        std::cout<<"init failed,please check!!"<<std::endl;
        return -1;
    }
    if (left_image.empty()||left_image.data==nullptr||right_image.empty()||right_image.data==nullptr)
    {
        std::cout<<"Image_src is empty!!!"<<std::endl;
        return -1;
    }
    RectifyImage(left_image,right_image);
    crestereo.RunCREStereo(left_image,right_image,pointcloud,disparity);
    return 0;    
}


int CREStereoAlgorithm::RunCREStereo(cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity)
{
    if (initflag!=0)
    {
        std::cout<<"init failed,please check!!"<<std::endl;
        return -1;
    }
    if (left_image.empty()||left_image.data==nullptr||right_image.empty()||right_image.data==nullptr)
    {
        std::cout<<"Image_src is empty!!!"<<std::endl;
        return -1;
    }
    crestereo.RunCREStereo(left_image,right_image,pointcloud,disparity);
    return 0;    
}

int CREStereoAlgorithm::Release()
{
    return crestereo.Release();
}

void CREStereoAlgorithm::ReadObjectYml(const char* filename, CalibrationParam&Calibrationparam)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["intrinsic_left"] >> Calibrationparam.intrinsic_left;
	fs["distCoeffs_left"] >> Calibrationparam.distCoeffs_left;
	fs["intrinsic_right"] >> Calibrationparam.intrinsic_right;
	fs["distCoeffs_right"] >> Calibrationparam.distCoeffs_right;
	fs["R"] >> Calibrationparam.R;
	fs["T"] >> Calibrationparam.T;
	fs["R_L"] >> Calibrationparam.R_L;
	fs["R_R"] >> Calibrationparam.R_R;
	fs["P1"] >> Calibrationparam.P1;
	fs["P2"] >> Calibrationparam.P2;
  	fs["Q"] >> Calibrationparam.Q;
	fs.release();
	return;
}

int CREStereoAlgorithm::RectifyImage(cv::Mat&Image_src,cv::Mat&rectifyImageL2,cv::Mat&rectifyImageR2)
{
	cv::Mat img_left, img_right;
    img_left = Image_src(cv::Range(0,480),cv::Range(0,640)).clone();
    img_right= Image_src(cv::Range(0,480),cv::Range(640,1280)).clone();
    //RectifyImage
    cv::Size s1, s2;
    s1 = img_left.size();
    s2 = img_right.size();
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(Calibrationparam.intrinsic_left, Calibrationparam.distCoeffs_left, Calibrationparam.R_L, Calibrationparam.P1, s1, CV_16SC2, mapLx, mapLy);
    cv::initUndistortRectifyMap(Calibrationparam.intrinsic_right, Calibrationparam.distCoeffs_right, Calibrationparam.R_R, Calibrationparam.P2, s1, CV_16SC2, mapRx, mapRy);
    cv::remap(img_left, rectifyImageL2, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(img_right, rectifyImageR2, mapRx, mapRy, cv::INTER_LINEAR);
	return 0;
}
int CREStereoAlgorithm::RectifyImage(cv::Mat&rectifyImageL2,cv::Mat&rectifyImageR2)
{
    //RectifyImage
    cv::Size s1, s2;
    s1 = rectifyImageL2.size();
    s2 = rectifyImageR2.size();
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(Calibrationparam.intrinsic_left, Calibrationparam.distCoeffs_left, Calibrationparam.R_L, Calibrationparam.P1, s1, CV_16SC2, mapLx, mapLy);
    cv::initUndistortRectifyMap(Calibrationparam.intrinsic_right, Calibrationparam.distCoeffs_right, Calibrationparam.R_R, Calibrationparam.P2, s1, CV_16SC2, mapRx, mapRy);
    //cv::Mat rectifyImageL2, rectifyImageR2;
    cv::remap(rectifyImageL2, rectifyImageL2, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(rectifyImageR2, rectifyImageR2, mapRx, mapRy, cv::INTER_LINEAR);
	return 0;
}

//Description   ModuleConfig           
//Params1       m_3DAlgorithmCallBack1                  
//Params2       config					
//Return		int						
//				other					
void* Initialize(char* model_path,int gpu_id,char*calibration_path)
{
    CREStereoAlgorithm*CREStereo_algorithm=new CREStereoAlgorithm();
    CREStereo_algorithm->Initialize(model_path,gpu_id,calibration_path);
    return CREStereo_algorithm;
}

//Description   
//Params		img			
//Return        int          
int RunCREStereo(void* p,cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity)
{
    CREStereoAlgorithm*CREStereo_algorithm=(CREStereoAlgorithm*)p;
    return CREStereo_algorithm->RunCREStereo(left_image,right_image,pointcloud,disparity);
}


int RunCREStereo_RectifyImage(void* p,cv::Mat&left_image,cv::Mat&right_image,float*pointcloud,cv::Mat&disparity)
{
    CREStereoAlgorithm*CREStereo_algorithm=(CREStereoAlgorithm*)p;
    return CREStereo_algorithm->RunCREStereo_RectifyImage(left_image,right_image,pointcloud,disparity);
}

//Description   
//Params		
//Return
const char* Version(void* p)
{
    return "CREStereoAlgorithm_V1.0";
}

//Description   
//Params		
//Return		int			
int Release(void* p)
{
    CREStereoAlgorithm*CREStereo_algorithm=(CREStereoAlgorithm*)p;
    return CREStereo_algorithm->Release();
}