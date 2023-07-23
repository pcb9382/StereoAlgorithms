# StereoAlgorithms
简体中文 | [English](./RAFTStereoAlgorithm_en.md)

如果觉得有用，不妨给个Star⭐️🌟支持一下吧~ 谢谢！

# Stereo_Calibration(双目相机标定)
## 使用方法
1. 首先使用process_image.py将1280*480图像分割成640*480，分割的图像保存在根目录下(我用的是双目相机)
2. 将对应生成的left*.jpg,right*.jpg图像名称放入stereo_calib.xml中，保证left,right顺序填写;
3. 运行标定软件Stereo_Calibration: $./Stereo_Calibration 5 8 40 1
    1. param1:程序名称
    2. param2:纵向内角点数
    3. param3:横向内角点数
    4. param4:棋盘格大小
    5. param5:是否显示标定过程中的的图像
4. 最终生成StereoCalibration.yml的标定文件

## 注意
1.  在标定显示的过程中，可以将角点检测不好的图像(一般是远处的角点比较小的)去除后重新标定


# RAFTStrereo
## 导出到onnx
1. 下载 [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo/tree/main)
2. 因为F.grid_sample op直到onnx 16才支持，这里转换为mmcv的bilinear_grid_sample的op;
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

# CREStereo(to do)

# DistDepth(to do)

# Hitnet(to do)

# RealtimeStereo(to do)


# 使用方法
## 1.模型下载
([Baidu Drive](链接: https://pan.baidu.com/s/1tgeqPmjPeKmCDQ2NGJZMWQ code: hdiv))
| 模型 |  作用    |  说明   |
|:----------|:----------|:----------|
|raftstereo-sceneflow_480_640_poly.onnx   |sceneflow双目深度估计模型|        
|raftstereo-realtime_480_640_ploy.onnx	   |realtime双目深度估计|             

## 2.环境
1. ubuntu20.04+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
2. ubuntu18.04+cuda10.2+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
3. nano,TX2,TX2-NX,xvier-NX                       (测试通过)
4. 其他环境请自行尝试或者加群了解


## 3.编译

1. 更改根目录下的CMakeLists.txt,设置tensorrt的安装目录
```
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. 默认opencv已安装，cuda,cudnn已安装
3. 为了Debug默认编译 ```-g O0``` 版本,如果为了加快速度请编译Release版本

4. 使用Visual Studio Code快捷键编译(4,5二选其一):
```
   ctrl+shift+B
```
5. 使用命令行编译(4,5二选其一):
```
   mkdir build
   cd build
   cmake ..
   make -j6
```
 

# References
1. https://github.com/princeton-vl/RAFT-Stereo
2. https://github.com/nburrus/RAFT-Stereo

# Acknowledgments & Contact 
## 1.WeChat ID: cbp931126
加我微信(备注：StereoAlgorithm),拉你进群
## 2.QQ Group：517671804
