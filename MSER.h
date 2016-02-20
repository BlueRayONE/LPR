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
	const float MAX_HEIGHT_SCALE = 1.5f;			//
	const float MAX_WIDTH_SCALE = 2.0f;				// same as MAX_HEIGHT_SCALE but for rect width
	const float MAX_BBOX_HEIGHT_SCALE = 3.0f;			// 
	const float MAX_RADIENT_ALLOWED = 0.35f;		// approx. 20 degree
	const float MAX_PART_OUTLIERS_ALLOWED = 0.15f;	//equals 10 percent outlier
	const uint MIN_DISTANCE_OUTLIER = 5;			//min distance point to line to be marked as outlier in pixel


	std::pair<cv::Mat, std::vector<cv::Rect>> mserFeature(cv::Mat grey, bool plus = true);
	//only for performance
	std::vector<cv::Rect> preDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);

	//real discard
	std::vector<cv::Rect> realDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);
	std::vector<cv::Rect> discardBBoxes_m(std::vector<cv::Rect>);

	std::tuple<bool, float, float> sameSize(std::vector<cv::Rect> innerElements);
};

#endif