#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
	Mat srcImg = imread("E:\\研究生资料\\老婆\\相册\\145kb.jpg");
	if (!srcImg.data)
		return -1;
	Mat hsvImg;
	cvtColor(srcImg, hsvImg, COLOR_BGR2HSV);
	Mat hueImg;
	hueImg.create(hsvImg.size(), hsvImg.depth());
	int ch[] = { 0,0 };
	mixChannels(&hsvImg, 1, &hueImg, 1, ch, 1);
	int bins = 25;
	MatND hist;
	int histSize = MAX(bins, 2);
	float hue_range[] = { 0,100 };
	const float* ranges = { hue_range };
	calcHist(&hueImg, 1, 0, Mat(),
		hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX,
		-1, Mat());
	MatND backproj;
	calcBackProject(&hueImg, 1, 0, hist, backproj,
		&ranges, 1, true);
	int w = 320;
	int h = 360;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);
	for (int i = 0; i < bins; i++)
	{
		rectangle(histImg, Point(i * bin_w, h),
			Point((i + 1) * bin_w,
				h - cvRound(hist.at<float>(i) * h / 255.0)),
			Scalar(0, 0, 255), -1);
	}
	imshow("BackProj", backproj);
	imshow("srcImg", srcImg);
	imshow("Histogram", histImg);
	equalizeHist(backproj, backproj);
	imshow("backproj_equa", backproj);
	waitKey(0);
	return 0;
}