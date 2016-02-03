#ifndef Wavelet_H
#define Wavelet_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>
#include <vector>

#include "ImageViewer.h"
#include <math.h>


const bool debug = true;

class Wavelet
{
public:
	Wavelet();
	~Wavelet();
	void run(cv::Mat img);

private:
	const float HEIGHT_TO_WIDTH_RATIO = 5.0f;
	const float GAP_TO_HEIGHT_RATIO = 68.0f / 220;
	const float AVG_WEIGHT = 0.5f; //1.1f;
	const float MAX_RECT_HEIGHT_RATIO = 110.0f / 1000; //120
	const float MIN_RECT_HEIGHT_RATIO = 20.0f / 1000; //50
	const float WEIGHT_STEP_SIZE = 0.025f;
	const bool DISCARD_EXCEEDED_RECT = true;	
	const int DEBUG_LEVEL = 0; //0 only results, 1 half debug, 2 full debug


	cv::Mat genGreyScale(cv::Mat img);
	cv::Mat haarWavelet(cv::Mat &src, int NIter);

	cv::Mat binarize(cv::Mat img);
	cv::Mat morph(cv::Mat img);

	void filterNeighbours(cv::Mat img);

	std::vector<int> calcRowSums(cv::Mat img);
	std::vector<int> calcColSums(cv::Mat img);

	std::vector<std::pair<int, float>> findPeaks(std::vector<float> arr);
	std::pair<int, float> findMaxPeak(std::vector<std::pair<int, float>> peakIDs, std::vector<float> arr);
	std::vector<float> gaussFilter(std::vector<int> arr);
	std::vector<float> movingAvg(std::vector<float> arr);

	float rectRank(cv::Mat img, cv::Rect rect);
	std::vector<std::pair<int, int>> findThresholdAreas(double avg, std::vector<float>, bool splitAreas = true);
	std::vector<std::pair<float, cv::Rect>> findRoughCandidate(cv::Mat img, std::vector<std::pair<int, int>> startRowsHeights);
	std::vector<std::pair<float, cv::Rect>> findNonIntCandidate(std::vector<std::pair<float, cv::Rect>> candidates);
	std::vector<std::pair<float, cv::Rect>> findExactCandidate(cv::Mat grey, cv::Mat rankImg, std::vector<std::pair<float, cv::Rect>> candidates);
	
	bool rectIntersect(cv::Rect r1, cv::Rect r2);
	bool evalRect(cv::Rect rect, float rank, cv::Mat evalImg);
	cv::Mat gbrHist(cv::Mat img);

	template<typename T> void print(std::vector<T> arr);
};


#endif

