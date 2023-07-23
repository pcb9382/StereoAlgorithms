
"""
功能:将1280*480分割成两张640*480的左右图像
作者:pcb
时间:2022.8.2
"""
import os
import cv2

#原始图像路径
imgaepath="Stereo_Raw/"

#分割图像保存的路径
save_image="left_right_image/"
imagelist=os.listdir(imgaepath)
i=0
for img_name in imagelist:
    raw_image=cv2.imread(imgaepath+img_name)
    left_image=raw_image[:,0:640]
    right_image=raw_image[:,640:1280]
    #cv2.imshow("left",left_image)
    #cv2.imshow("right",right_image)
    #cv2.waitKey(0)
    image_left_name=save_image+"left"+str(i)+".jpg"
    image_right_name = save_image + "right" + str(i) + ".jpg"
    cv2.imwrite(image_left_name,left_image)
    cv2.imwrite(image_right_name,right_image)
    i=i+1


