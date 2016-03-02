#ifndef MSER_H
#define MSER_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <QDebug>
#include <vector>

#include "ImageViewer.h"
#include <math.h>


class MSER
{
public:
	MSER(cv::Mat imgOrig);
	~MSER();
	std::vector<cv::Rect> run();
    static std::pair<cv::Mat, std::vector<cv::Rect>> mserFeature(cv::Mat grey, bool plus = true);
	std::tuple<double, double, double, double> meanStdDev(std::vector<cv::Rect> elems);
private:
	const float MAX_HEIGHT_SCALE = 1.5f;			//
	const float MAX_WIDTH_SCALE = 2.0f;				// same as MAX_HEIGHT_SCALE but for rect width
	const float MAX_BBOX_HEIGHT_SCALE = 3.5f;		// 3.0f
	const float MAX_RADIENT_ALLOWED = 0.4f;			// approx. 22 degree
	const float MAX_PART_OUTLIERS_ALLOWED = 0.15f;	//equals 15 percent outlier
	const uint MIN_DISTANCE_OUTLIER = 5;			//min distance point to line to be marked as outlier in pixel
	const float MAX_ASPECT_RATIO = 5.5f;			//eu license plate are 52cm by 11cm --> 52/11 = 4.72727272
	const float MIN_ASPECT_RATIO = 1.1f;
	const uint RELAX_PIXELS = 10;
	const int DEBUG_LEVEL = 1;

	cv::Mat resizeImg(cv::Mat img);
	std::vector<std::pair<cv::Rect, int>> getNumInnerElements(std::vector<cv::Rect>, std::vector<cv::Rect>);

	std::vector<cv::Rect> preDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);
	std::vector<cv::Rect> realDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);
	std::vector<cv::Rect> postDiscardBBoxes_p(std::vector<cv::Rect>, std::vector<cv::Rect>);

	std::tuple<bool, float, float> sameSize(std::vector<cv::Rect> innerElements);

	cv::Rect relaxRect(cv::Rect rect);
	cv::Mat getROI(cv::Rect rect);
	int intersectArea(cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };


	double scaleFactor;
	cv::Mat originalImage;
	cv::Mat resizedImage;
	cv::Mat grey, mser_p, mser_m;
	//only for visualiztation
	cv::Mat visualize_p, colorP, colorP2, colorP3, colorM, img_bk;

};

#endif
