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
	float GAP_TO_HEIGHT_RATIO = 68.0f / 220;
	float AVG_WEIGHT = 1.1f;
	float MAX_RECT_HEIGHT_RATIO = 150.0f / 1000;
	float MIN_RECT_HEIGHT_RATIO = 20.0f / 1000;
	const bool DISCARD_EXCEEDED_RECT = true;	
	const int DEBUG_LEVEL = 0; //0 only results, 1 half debug, 2 full debug

	

	cv::Mat genGreyScale(cv::Mat img);

	cv::Mat haarWavelet(cv::Mat &src, int NIter);
	double* daubTrans(double*, int);

	cv::Mat binarize(cv::Mat img);
	cv::Mat morph(cv::Mat img);

	int* calcRowSums(cv::Mat img);
	int* calcColSums(cv::Mat img);

	std::vector<std::pair<int, float>> findPeaks(float* arr, int n);
	std::pair<int, float> findMaxPeak(std::vector<std::pair<int, float>> peakIDs, float* arr);
	float* gaussFilter(int* arr, int n);
	float* movingAvg(float* arr, int n);

	float rectRank(cv::Mat img, cv::Rect rect);
	std::vector<std::pair<int, int>> findThresholdAreas(int n, double avg, float* rowSums);
	std::vector<std::pair<float, cv::Rect>> findRoughCandidate(cv::Mat img, std::vector<std::pair<int, int>> startRowsHeights);
	std::vector<std::pair<float, cv::Rect>> findNonIntCandidate(std::vector<std::pair<float, cv::Rect>> candidates, int n);
	std::vector<std::pair<float, cv::Rect>> findExactCandidate(cv::Mat grey, cv::Mat rankImg, std::vector<std::pair<float, cv::Rect>> candidates);
	
	bool rectIntersect(cv::Rect r1, cv::Rect r2);

	bool evalRect(cv::Rect rect, float rank, cv::Mat evalImg);

	template<typename T> void print(T* arr, int n);
};


#endif

