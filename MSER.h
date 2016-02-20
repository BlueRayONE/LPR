#ifndef MSER_H
#define MSER_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <QDebug>
#include <vector>

#include "ImageViewer.h"
#include <math.h>


class MSER
{
public:
	MSER();
	~MSER();
	void run(cv::Mat img);
private:
	const float MAX_HEIGHT_SCALE = 1.5f;
	const float MAX_WIDTH_SCALE = 1.5f;
	const float MAX_BBOX_HEIGHT_SCALE = 2;
	//const uint MAX_BBOX_WIDTH_VARIANCE = 10;

	//only for performance
	std::vector<cv::Rect> preDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);

	//real discard
	std::vector<cv::Rect> realDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);
	std::vector<cv::Rect> discardBBoxes_m(std::vector<cv::Rect>);

	std::tuple<bool, float, float> sameSize(std::vector<cv::Rect> innerElements);
};

#endif