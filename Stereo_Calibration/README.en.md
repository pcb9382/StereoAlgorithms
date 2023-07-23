# DecodeAlgorithm

#### Description
双目相机标定程序

#### 使用方法
1. 首先使用process_image.py将1280*480图像分割成640*480，分割的图像保存在根目录下
2. 将对应生成的left*.jpg,right*.jpg图像名称放入stereo_calib.xml中，保证left,right顺序填写;
3. 运行标定软件Stereo_Calibration: $./Stereo_Calibration 5 8 40 1
    1)param1:程序名称
    2)param2:纵向内角点数
    3)param3:横向内角点数
    4)param4:棋盘格大小
    5)param5:是否显示标定过程中的的图像
4. 最终生成StereoCalibration.yml的标定文件

#### 注意

1.  在标定显示的过程中，可以将角点检测不好的图像(一般是远处的角点比较小的)去除后重新标定
2.  
3.  xxxx

#### Instructions

1.  xxxx
2.  xxxx
3.  xxxx

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
