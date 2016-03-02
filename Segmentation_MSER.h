#ifndef SEGMENTATION_MSER_H
#define SEGMENTATION_MSER_H

#include "opencv2/opencv.hpp"
#include <vector>
#include "MSER.h"
#include "ImageViewer.h"

class Segmentation_MSER
{
public:
	Segmentation_MSER(cv::Mat img);
    std::vector<cv::Mat> findChars();
private:
	const float DERIV = 1.0/8.0; // DERIV * rows distance from fitted line and from other centers are classified as inliers
	const float DIST_TO_MEAN = 2.0f;
	const int RELAX_PIXELS_VERT = 10;
	const int RELAX_PIXELS_HOR = 5;
	const float MAX_ALLOWED_HEIGHT_DEV = 0.3f; //30 percent
	const float MAX_ALLOWED_WIDTH_DEV = 0.75f; //75 percent
	const int DEBUG_LEVEL = 1; //0 no output, 1 mser with bbox, 2 all digits

	cv::Mat originalImage;

	std::pair< cv::Mat, std::vector<cv::Rect>> mserFeature(cv::Mat grey);
	std::vector<cv::Rect> discardOverlapping(std::vector<cv::Rect> bbox);
	std::vector<cv::Rect> discardOutlier(std::vector<cv::Rect> bbox);
	cv::Rect relaxRect(cv::Rect rect);	
};

#endif // SEGMENTATION_MSER_H
