## 1.三维重建效果
   <img src="https://media.giphy.com/media/nYqxbmAdGDgVJ2lQYK/giphy.gif" alt="drawing" width="380"/> <img src="https://media.giphy.com/media/y8hD5SNh1QHc8yCGBv/giphy.gif" alt="drawing" width="380"/>

## 2.pth导出到onnx
1. 下载 [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo/tree/main)
2. 因为F.grid_sample op直到onnx 16才支持，这里转换为mmcv的bilinear_grid_sample的op
   
   1)需要安装mmcv;
   
   2)F.grid_sample替换为bilinear_grid_sample;
3. 导出onnx模型
   
   1） 导出sceneflow模型
   ```
   （1）python3 export_onnx.py --restore_ckpt models/raftstereo-sceneflow.pth
   （2）onnxsim raftstereo-sceneflow_480_640.onnx raftstereo-sceneflow_480_640_sim.onnx
   （3）(option)polygraphy surgeon sanitize --fold-constants raftstereo-sceneflow_480_640_sim.onnx -o raftstereo-sceneflow_480_640_sim_ploy.onnx
   ```
   2）导出realtime模型
   ```
   （1）python3 export_onnx.py --restore_ckpt models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --mixed_precision
   
   （2）onnxsim raftstereo-realtime_480_640.onnx raftstereo-realtime_480_640_sim.onnx

   （3）(option)polygraphy surgeon sanitize --fold-constants raftstereo-realtime_480_640_sim.onnx -o raftstereo-realtime_480_640_sim_ploy.onnx
   ```

## 3.使用导出的onnx模型或者下载已经转好的onnx模型
### 1.模型下载
([Baidu Drive](链接: https://pan.baidu.com/s/1tgeqPmjPeKmCDQ2NGJZMWQ code: hdiv))

### 2.参数设置(最好写绝对路径或者将需要的文件拷贝到build目录下)
```
   //双目相机标定文件
   char* stereo_calibration_path="StereoCalibration.yml";
   //onnx模型路径，自动将onnx模型转为engine模型
   char* strero_engine_path="raftstereo-sceneflow_480_640_poly.onnx"; 
   //相机采集的左图像
   cv::Mat imageL=cv::imread("left0.jpg");
   //相机采集的右图像
   cv::Mat imageR=cv::imread("right0.jpg");
```
### 4.RAFTStereo模块编译运行(确保已经将step2中需要的文件拷贝到build文件夹下)
   ```
   cd RAFTStereo
   mkdir build&&cd build
   cmake ..&&make -j8
   ./raft_stereo_demo
   ```
### 5.运行结果
   1. 会在运行目录下保存视差图disparity.jpg
   2. 会在运行目录下保存pointcloud.txt文件，每一行表示为x,y,z,r,g,b
   
   <img src="../resource/left0.jpg" alt="drawing" width="380"/> <img src="../resource/right0.jpg" alt="drawing" width="380"/>
   <img src="../resource/disparity.jpg" alt="drawing" width="380"/><img src="../resource/pointcloud+rgb.png" alt="drawing" width="380"/>
    
### 6.模型说明
| 模型 |  说明   |  备注 |
|:----------:|:----------:|:----------|
|raftstereo-sceneflow_480_640_poly.onnx   |sceneflow双目深度估计模型| ([Baidu Drive](链接: https://pan.baidu.com/s/1tgeqPmjPeKmCDQ2NGJZMWQ code: hdiv)) |     
|raftstereo-realtime_480_640_ploy.onnx	   |realtime双目深度估计模型| 可自行下载模型进行转化|   

### 7.其他
  平台|  sceneflow(640*480)耗时  |realtime(640*480)耗时|说明|
|:----------:|:----------:|:----------:|:----------:|
|3090|38ms| 11ms ||   
|3060|83ms|24ms| ||
|jetson Xavier-NX||120ms|sceneflow未尝试|
|jetson TX2-NX||400ms|sceneflow未尝试|
|jetson Nano|||支持|