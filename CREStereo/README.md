# CREStereo
## 1.三维重建效果
### 1.Stereo depth estimation
   <img src="../resource/teaser.jpg" alt="drawing" width="800"/>

### 2.left_image,right_image,heat_map
   <img src="../resource/left.png" alt="drawing" width="250"/> <img src="../resource/right.png" alt="drawing" width="250"/><img src="../resource/output.jpg" alt="drawing" width="250"/>

## 2.使用导出的onnx模型
### 1.模型下载
([Baidu Drive](链接: https://pan.baidu.com/s/1lGL8FOjcy6c1y5oDJLYA4w 提取码: gimg))

### 2.参数设置(最好写绝对路径或者将需要的文件拷贝到/build/CREStereo/test/crestereo_demo目录下)
```
   //双目相机标定文件
   char* stereo_calibration_path="StereoCalibration.yml";
   //onnx模型路径，自动将onnx模型转为engine模型
   char* strero_engine_path="crestereo_init_iter10_480x640.onnx"; 
   //相机采集的左图像
   cv::Mat imageL=cv::imread("left0.jpg");
   //相机采集的右图像
   cv::Mat imageR=cv::imread("right0.jpg");
```
### 3.编译运行(确保已经将step2中需要的文件拷贝到相应的文件夹下)
   ```
   cd StereoAlgorithms
   mkdir build&&cd build
   cmake ..&&make -j8
   ./build/CREStereo/test/crestereo_demo
   ```
### 4.运行结果
   1. 会在运行目录下保存视差图disparity.jpg;
   2. 会在运行目录下保存pointcloud.txt文件，每一行表示为x,y,z,r,g,b;
   3. 会在运行目录下保存heatmap.jpg热力图;
   
   <img src="../resource/left0.jpg" alt="drawing" width="380"/> <img src="../resource/CREStereo_disparity.jpg" alt="drawing" width="380"/>
   <img src="../resource/CREStereo_heatmap.jpg" alt="drawing" width="380"/><img src="../resource/CREStereo_pointcloud.png" alt="drawing" width="380"/>
    
### 6.其他
  平台|iter2_480x640|iter5_480x640|iter10_480x640|说明|
|:----------:|:----------:|:----------:|:----------:|:----------:|
|3090|12ms|23ms|42ms||   
|3060||||未测试|
|jetson Xavier-NX||||未测试|
|jetson TX2-NX||||未测试|
|jetson Nano||||未测试|