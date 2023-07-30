5.FastACVNet_plus(,只支持TensoRT8.6+！！！)
## 1.三维重建效果
### 1.Demo on KITTI raw data
   <img src="../resource/kittiraw_demo.gif" alt="drawing" width="800"/>

### 2.Qualitative results on Scene Flow
   <img src="../resource/sceneflow.png" alt="drawing" width="800"/> 

## 2.使用导出的onnx模型
### 1.模型下载
([Baidu Drive](链接: https://pan.baidu.com/s/1kxrNLlAFgpTwECF21SM9_g 提取码: 83qn))

### 2.参数设置(最好写绝对路径或者将需要的文件拷贝到/build/FastACVNet_plus/test/文件夹下)
```
   //双目相机标定文件
   char* stereo_calibration_path="StereoCalibration.yml";
   //onnx模型路径，自动将onnx模型转为engine模型
   char* strero_engine_path="fast_acvnet_plus_generalization_opset16_480x640.onnx"; 
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
   ./build/FastACVNet_plus/test/fastacvnet_plus_demo
   ```
### 4.运行结果
   1. 会在运行目录下保存视差图disparity.jpg;
   2. 会在运行目录下保存pointcloud.txt文件，每一行表示为x,y,z,r,g,b;
   3. 会在运行目录下保存heatmap.jpg热力图;
   
   <img src="../resource/left0.jpg" alt="drawing" width="360"/> <img src="../resource/FastACVNet_plus_disparity.jpg" alt="drawing" width="360"/>
   <img src="../resource/FastACVNet_plus_heatmap.jpg" alt="drawing" width="360"/><img src="../resource/FastACVNet_plus_pointcloud.png" alt="drawing" width="360"/>
    
### 6.其他
  平台|generalization_opset16_480x640|说明|
|:----------:|:----------:|:----------:|
|3090|12ms||   
|3060||未测试|
|jetson Xavier-NX||未测试|
|jetson TX2-NX||未测试|
|jetson Nano||未测试|

### References
1. https://github.com/gangweiX/Fast-ACVNet
2. https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation/tree/main
3. https://github.com/ibaiGorordo/ONNX-ACVNet-Stereo-Depth-Estimation
4. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/338_Fast-ACVNet