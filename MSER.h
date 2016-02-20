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
	std::vector<cv::Rect> run(cv::Mat img);
private:
	const float MAX_HEIGHT_SCALE = 1.5f;			//
	const float MAX_WIDTH_SCALE = 2.0f;				// same as MAX_HEIGHT_SCALE but for rect width
	const float MAX_BBOX_HEIGHT_SCALE = 3.5f;		// 3.0f
	const float MAX_RADIENT_ALLOWED = 0.35f;		// approx. 20 degree
	const float MAX_PART_OUTLIERS_ALLOWED = 0.15f;	//equals 15 percent outlier
	const uint MIN_DISTANCE_OUTLIER = 5;			//min distance point to line to be marked as outlier in pixel
	const float MAX_ASPECT_RATIO = 5.0f;			//eu license plate are 52cm by 11cm --> 52/11 = 4.72727272
	const uint RELAX_PIXELS = 5;


	std::pair<cv::Mat, std::vector<cv::Rect>> mserFeature(cv::Mat grey, bool plus = true);
	std::vector<std::pair<cv::Rect, int>> getInnerElements(std::vector<cv::Rect>, std::vector<cv::Rect>);


	//only for performance
	std::vector<cv::Rect> preDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);

	//real discard
	std::vector<cv::Rect> realDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);

	std::vector<cv::Rect> postDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);
	std::vector<cv::Rect> discardBBoxes_m(std::vector<cv::Rect>);

	std::tuple<bool, float, float> sameSize(std::vector<cv::Rect> innerElements);
};

#endif