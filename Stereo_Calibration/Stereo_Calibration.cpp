
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <cstdio>
#include <stdlib.h>
struct  CalibrationParam
{
	cv::Mat intrinsic_left;
	cv::Mat distCoeffs_left;
	cv::Mat	intrinsic_right;
	cv::Mat distCoeffs_right;
	cv::Mat R;
	cv::Mat T;
	cv::Mat R_L;
	cv::Mat R_R;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;
	cv::Rect validROIL, validROIR;
};



void WriteObjectYml(const char* filename, const char* variablename, const cv::Mat &source)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << variablename << source;
	fs.release();
}

void ReadObjectYml(const char* filename, CalibrationParam&Calibrationparam)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["validROIL"] >> Calibrationparam.validROIL;
	fs["validROIR"] >> Calibrationparam.validROIR;
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
	fs.release();
}
static bool readStringList(const std::string& filename, std::vector<std::string>& l)
{
	l.resize(0);
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
		return false;
	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((std::string)*it);
	return true;
}
void StereoCalibration(std::vector<std::string>imagelist, int numCornersVer, int numCornersHor, int numSquares ,int ShowChessCorners)
{
	if (imagelist.size() % 2 != 0)
	{
		std::cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	std::vector<std::vector<cv::Point2f>> image_leftPoints, image_rightPoints;
	std::vector<std::vector<cv::Point3f>> objectPoints;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
	cv::Mat gray_l, gray_r;
	cv::Mat image_l, image_r;
	std::vector<cv::Point3f> obj;
	for (int i = 0; i < numCornersHor; i++)
	{
		for (int j = 0; j < numCornersVer; j++)
		{
			obj.push_back(cv::Point3f((float)j * numSquares, (float)i * numSquares, 0));
		}	
	}
	cv::Size s1, s2;
	for (int i = 0; i < imagelist.size()/2; i++)
	{
		
		image_l = cv::imread(imagelist[2*i]);
		image_r = cv::imread(imagelist[2*i+1]);
		s1 = image_l.size();
		s2 = image_r.size();

		cvtColor(image_l, gray_l, cv::COLOR_BGR2GRAY);
		cvtColor(image_r, gray_r, cv::COLOR_BGR2GRAY);
		std::vector<cv::Point2f> corners_r;
		std::vector<cv::Point2f> corners_l;
		bool ret1 = findChessboardCorners(gray_r, cv::Size(numCornersVer, numCornersHor), corners_r, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
		bool ret2 = findChessboardCorners(gray_l, cv::Size(numCornersVer, numCornersHor), corners_l, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
		if (ret1&&ret2&&ShowChessCorners)
		{
			cornerSubPix(gray_l, corners_l, cv::Size(5, 5), cv::Size(-1, -1), criteria);
			drawChessboardCorners(image_l, cv::Size(numCornersVer, numCornersHor), corners_l, ret1);
			imshow("ChessboardCorners", image_l);
			std::cout << imagelist[2 * i] << std::endl;
			cv::waitKey(0);
			cornerSubPix(gray_r, corners_r, cv::Size(5, 5), cv::Size(-1, -1), criteria);
			drawChessboardCorners(image_r, cv::Size(numCornersVer, numCornersHor), corners_r, ret2);
			imshow("ChessboardCorners", image_r);
			std::cout << imagelist[2 * i+1] << std::endl;
			cv::waitKey(0);
		}
		if (ret1&&ret2)
		{
			image_rightPoints.push_back(corners_r);
			image_leftPoints.push_back(corners_l);
			objectPoints.push_back(obj);
		}

	}
	if (ShowChessCorners)
	{
		cv::destroyWindow("ChessboardCorners");
	}
	
	cv::Mat intrinsic_left = cv::Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffs_left;
	std::vector<cv::Mat> rvecs_l;
	std::vector<cv::Mat> tvecs_l;

	intrinsic_left.ptr<float>(0)[0] = 1;
	intrinsic_left.ptr<float>(1)[1] = 1;
	cv::calibrateCamera(objectPoints, image_leftPoints, s1, intrinsic_left, distCoeffs_left, rvecs_l, tvecs_l);
	cv::Mat intrinsic_right = cv::Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffs_right;
	std::vector<cv::Mat> rvecs_r;
	std::vector<cv::Mat> tvecs_r;
	cv::Mat R_total;
	cv::Vec3d T_total;
	intrinsic_right.ptr<float>(0)[0] = 1;
	intrinsic_right.ptr<float>(1)[1] = 1;
	cv::calibrateCamera(objectPoints, image_rightPoints, s2, intrinsic_right, distCoeffs_right, rvecs_r, tvecs_r);
	cv::Mat R_L;
	cv::Mat R_R;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;
	cv::Rect validROIL, validROIR;
	cv::Mat E;
	cv::Mat F;
	cv::Mat R;
	cv::Mat T;
	std::cout << "Stereo Calibration start!" <<std::endl;
	double rms = cv::stereoCalibrate(objectPoints, image_leftPoints, image_rightPoints,intrinsic_left, distCoeffs_left,intrinsic_right, distCoeffs_right,
		cv::Size(1920,1080), R, T, E, F, cv::CALIB_USE_INTRINSIC_GUESS,cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));                                                                                                                        
	std::cout << "Stereo Calibration done with RMS error = " << rms << std::endl;

	std::cout << "Starting Rectification" << std::endl;
	stereoRectify(intrinsic_left, distCoeffs_left, intrinsic_right, distCoeffs_right, s1, R, T, R_L, R_R, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, s1, &validROIL, &validROIR);//
	std::cout << "Rectification Done" << std::endl;
	std::cout << "Save Calibration to StereoCalibration.yml" << std::endl;
	cv::FileStorage fs("StereoCalibration.yml", cv::FileStorage::WRITE);
	fs << "intrinsic_left" << intrinsic_left;
	fs << "distCoeffs_left" << distCoeffs_left;
	fs << "intrinsic_right" << intrinsic_right;
	fs << "distCoeffs_right" << distCoeffs_right;
	fs << "R" << R;
	fs << "T" << T;
	fs << "R_L" << R_L;
	fs << "R_R" << R_R;
	fs << "P1" << P1;
	fs << "P2" << P2;
	fs << "Q" << Q;
	fs << "validROIL" << validROIL;
	fs << "validROIR" << validROIR;
	fs.release();
	std::cout << "Done Calibration" << std::endl;
	return;
}


void ShowMatchResult(cv::Mat&srcImg, std::vector<cv::KeyPoint>&srcKeypoint, cv::Mat&dstImg,
	std::vector<cv::KeyPoint>&dstKeypoint, std::vector<cv::DMatch>&goodMatch)
{
	cv::drawKeypoints(srcImg, srcKeypoint, srcImg);
	cv::drawKeypoints(dstImg, dstKeypoint, dstImg);

	cv::Mat img_matches;
	cv::drawMatches(srcImg, srcKeypoint, dstImg, dstKeypoint, goodMatch, img_matches,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches", img_matches);
	cv::imwrite("img_matches.jpg", img_matches);
	std::cout << "Good MatchPoint Num is :" << goodMatch.size() << std::endl;
	cv::waitKey(0);
	return;

}
void StitcingImages(cv::Mat &srcImg1, cv::Mat&srcImg2, std::string ImageName)
{

	cv::Mat img_merge;
	cv::Size size(srcImg1.cols + srcImg2.cols, MAX(srcImg1.rows, srcImg2.rows));
	img_merge.create(size, CV_MAKETYPE(srcImg1.depth(), 3));
	img_merge = cv::Scalar::all(0);
	cv::Mat outImg_left, outImg_right;
	outImg_left = img_merge(cv::Rect(0, 0, srcImg1.cols, srcImg1.rows));
	outImg_right = img_merge(cv::Rect(srcImg1.cols, 0, srcImg2.cols, srcImg2.rows));
	srcImg1.copyTo(outImg_left);
	srcImg2.copyTo(outImg_right);
	//namedWindow("image1", 0);
	//imshow("image1", img_merge);
	//waitKey();
	cv::imwrite(ImageName, img_merge);
	return;

}
int main(int argc,char** argv)
{
	int numCornersHor = 8; // atoi(argv[1]);// 8;	//水平
	int numCornersVer =11;//atoi(argv[2]);// 11;		//垂直
	int numSquares =25;//atoi(argv[3]);// 25;
	std::string rectifyImageSavePath = "Stereo_Calibration/rectifyImage";
	std::string imagelistfn="stereo_calib.xml";
	std::vector<std::string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		std::cout << "can not open " << imagelistfn << " or the string list is empty" << std::endl;
	}
	StereoCalibration(imagelist, numCornersVer, numCornersHor, numSquares,true);//atoi(argv[4])
	CalibrationParam Calibrationparam;
	ReadObjectYml("StereoCalibration.yml", Calibrationparam);
	for (int i = 0; i < imagelist.size() / 2; i++)
	{
		cv::Mat image_l = cv::imread(imagelist[i*2]);
		cv::Mat image_r = cv::imread(imagelist[i * 2+1]);
		cv::Size s1, s2;
		s1 = image_l.size();
		s2 = image_r.size();
		cv::Mat mapLx, mapLy, mapRx, mapRy;
		cv::initUndistortRectifyMap(Calibrationparam.intrinsic_left, Calibrationparam.distCoeffs_left, Calibrationparam.R_L, 
			Calibrationparam.P1, s1, CV_16SC2, mapLx, mapLy);
		cv::initUndistortRectifyMap(Calibrationparam.intrinsic_right, Calibrationparam.distCoeffs_right, Calibrationparam.R_R, 
			Calibrationparam.P2, s1, CV_16SC2, mapRx, mapRy);
		cv::Mat rectifyImageL2, rectifyImageR2;
		cv::remap(image_l, rectifyImageL2, mapLx, mapLy, cv::INTER_LINEAR);
		cv::remap(image_r, rectifyImageR2, mapRx, mapRy, cv::INTER_LINEAR);
		std::string Imagename_L = rectifyImageSavePath + imagelist[i * 2];
		std::string Imagename_R = rectifyImageSavePath + imagelist[i * 2+1];
		cv::imwrite(Imagename_L, rectifyImageL2);
		cv::imwrite(Imagename_R, rectifyImageR2);
		if(true)//atoi(argv[4])
		{
			cv::imshow("rectifyImageL", rectifyImageL2);
			cv::imshow("rectifyImageR", rectifyImageR2);
			cv::waitKey(2);
		}
		
		cv::Mat canvas;
		double sf;
		int w, h;
		sf = 1080. / MAX(s1.width, s1.height);
		w = cvRound(s1.width * sf);
		h = cvRound(s1.height * sf);
		canvas.create(h, w * 2, CV_8UC3);
		cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));
		cv::resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA); 
		cv::Rect vroiL(cvRound(Calibrationparam.validROIL.x*sf), cvRound(Calibrationparam.validROIL.y*sf), 
			cvRound(Calibrationparam.validROIL.width*sf), cvRound(Calibrationparam.validROIL.height*sf));
		cv::rectangle(canvasPart, vroiL, cv::Scalar(0, 0, 255), 3, 8);  
		canvasPart = canvas(cv::Rect(w, 0, w, h));   
		cv::resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR);
		cv::Rect vroiR(cvRound(Calibrationparam.validROIR.x * sf), cvRound(Calibrationparam.validROIR.y*sf),
			cvRound(Calibrationparam.validROIR.width * sf), cvRound(Calibrationparam.validROIR.height * sf));
		cv::rectangle(canvasPart, vroiR, cv::Scalar(0, 255, 0), 3, 8);
		for (int i = 0; i < canvas.rows; i += 40)
		{
			cv::line(canvas, cv::Point(0, i), cv::Point(canvas.cols, i), cv::Scalar(0, 255, 0), 1, 8);
		}
		if(1)//atoi(argv[4])
		{
			cv::imshow("rectified", canvas);
			cv::waitKey(0);
		}
	}
	return 0;
}

