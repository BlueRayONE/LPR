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
};

#endif // SEGMENTATION_MSER_H
