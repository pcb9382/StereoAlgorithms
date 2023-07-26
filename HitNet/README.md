## 1.三维重建效果
### 1.Stereo depth estimation
   <img src="../resource/out.jpg" alt="drawing" width="800"/>
   
   Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)
### 2.onnxHitnetDepthEstimation
   <img src="../resource/onnxHitnetDepthEstimation.gif" alt="drawing" width="800"/>

## 2.使用导出的onnx模型
### 1.模型下载
([Baidu Drive](链接: 链接: https://pan.baidu.com/s/1R3KU-pGJUJvGVOg8MPg8Nw 提取码: 6stm))

### 2.参数设置(最好写绝对路径或者将需要的文件拷贝到build目录下)
```
   //双目相机标定文件
   char* stereo_calibration_path="StereoCalibration.yml";
   //onnx模型路径，自动将onnx模型转为engine模型
   char* strero_engine_path="model_float32.onnx"; 
   //相机采集的左图像
   cv::Mat imageL=cv::imread("left0.jpg");
   //相机采集的右图像
   cv::Mat imageR=cv::imread("right0.jpg");
```
### 3.HitNet模块编译运行(确保已经将step2中需要的文件拷贝到build文件夹下)
   ```
   cd HitNet
   mkdir build&&cd build
   cmake ..&&make -j8
   ./HitNet_demo
   ```
### 4.运行结果
   1. 会在运行目录下保存视差图disparity.jpg
   2. 会在运行目录下保存pointcloud.txt文件，每一行表示为x,y,z,r,g,b
   
   <img src="../resource/left0.jpg" alt="drawing" width="380"/> <img src="../resource/right0.jpg" alt="drawing" width="380"/>
   <img src="../resource/disparity_HitNet.jpg" alt="drawing" width="380"/><img src="../resource/HitNet.png" alt="drawing" width="380"/>
    
### 6.其他
  平台|  middlebury_d400(640*480)耗时  |flyingthings_finalpass_xl(640*480)耗时|说明|
|:----------:|:----------:|:----------:|:----------:|
|3090|15ms|||   
|3060|||未测试|
|jetson Xavier-NX|||未测试|
|jetson TX2-NX|||未测试|
|jetson Nano|||未测试|