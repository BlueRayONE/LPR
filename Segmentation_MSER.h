#ifndef SEGMENTATION_MSER_H
#define SEGMENTATION_MSER_H

#include "opencv2/opencv.hpp"
#include <vector>
#include "MSER.h"
#include "ImageViewer.h"

class Segmentation_MSER
{
public:
    Segmentation_MSER();
    static std::vector<cv::Mat> findChars(cv::Mat img);
private:
	static std::vector<cv::Rect> discardOverlapping(std::vector<cv::Rect> bbox);
	static std::vector<cv::Rect> discardOutlier(std::vector<cv::Rect> bbox);
	static cv::Rect relaxRect(cv::Rect rect, int rows, int cols);
	static cv::Mat morph(cv::Mat img);
	static std::pair< cv::Mat, std::vector<cv::Rect>> mserFeature(cv::Mat grey);
};

#endif // SEGMENTATION_MSER_H
