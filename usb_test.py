# -*- coding: utf-8 -*-
import cv2
import time
 
if __name__ == "__main__":
 
    cap = cv2.VideoCapture(0,cv2.CAP_V4L)#cv2.CAP_V4L
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if cap.isOpened():
        #window_handle = cv2.namedWindow("D435", cv2.WINDOW_AUTOSIZE)
        
        # 逐帧显示
        i=0
        print("Enter q to exit!")
        print("Enter k to save!")
        while True:#cv2.getWindowProperty("D435", 0) >= 0
            
            begin_time=time.time()
            ret_val, img = cap.read()
            end_time=time.time()
            #print("time:{}".format(end_time-begin_time))
            cv2.imshow("USB Camera", img)
            k=cv2.waitKey(1)
            #img=cv2.resize(img,dsize=(1280,480))
            
            if ( k== ord('q')):
                break
            elif(k == ord('k')):
                image_name=str(i)+".jpg"
                cv2.imwrite(image_name, img)
                print("the {} image saved!".format(i))
                i=i+1
                
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("打开摄像头失败")
