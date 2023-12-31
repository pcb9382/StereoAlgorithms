#include"HitNetAlgorithm.h"
#include<iostream>

#include<chrono>
int main()
{

    char* stereo_calibration_path="StereoCalibration.yml";
    char* strero_engine_path="/home/pcb/Algorithm/StereoAlgorithmsPytorch/Hitnet/resources/middlebury_d400/saved_model_480x640/model_float32.onnx";
    
 

    cv::Mat imageL=cv::imread("left0.jpg");
    cv::Mat imageR=cv::imread("right0.jpg");
    
    //init
    void * raft_stereo=Initialize(strero_engine_path,0,stereo_calibration_path);

    float*pointcloud=new float[imageL.cols*imageL.rows*6];
    cv::Mat disparity;
    for (size_t i = 0; i < 1000;i++)
    {
        cv::Mat imageL1=imageL.clone();
        cv::Mat imageR1=imageR.clone();
        //auto start = std::chrono::system_clock::now();
        RunHitNet(raft_stereo,imageL1,imageR1,pointcloud,disparity);
        //auto end = std::chrono::system_clock::now();
		//std::cout<<"time:"<<(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())<<"ms"<<std::endl;
    }
    cv::imwrite("disparity.jpg",disparity);
    std::fstream pointcloudtxt;
    pointcloudtxt.open("pointcloud.txt",std::ios::out);
    for (size_t i = 0; i < imageL.cols*imageL.rows*6; i+=6)
    {
        pointcloudtxt<<pointcloud[i]<<" "<<pointcloud[i+1]<<" "<<pointcloud[i+2]<<" "
        <<pointcloud[i+3]<<" "<<pointcloud[i+4]<<" "<<pointcloud[i+5]<<std::endl;
    }
    pointcloudtxt.close();
    Release(raft_stereo);
    delete []pointcloud;
    pointcloud=nullptr;
    return 0;
}