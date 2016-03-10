#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageViewer.h"
#include <iostream>

class PCA_Localisation
{
public:
	/* Konstruktor */
    PCA_Localisation();

	/* Destruktor */
    ~PCA_Localisation();

    cv::Mat getPlate(cv::Mat src);


private:

	double _scaleFactor;

	cv::Mat computeSobelImgX(cv::Mat src); 
	int xGradient(cv::Mat greyImg, int x, int y); 

	cv::Mat computeRegions(cv::Mat src); 
	int sameColor(cv::Mat src, int x, int y); 

	cv::Mat closeOperation3x3(cv::Mat src); 
	void identidy3x30(cv::Mat src, cv::Mat dst); 
	int mask3x30(cv::Mat src, int x, int y); 

	void mergeResults(cv::Mat c1, cv::Mat c2, cv::Mat c3, cv::Mat dst); 
	

	cv::Mat findPlate(cv::Mat original, cv::Mat src, std::vector<std::vector<cv::Point>> contours); 
	int getBestScore(std::vector<std::vector<double>> data); 

	double getEigenVector(std::vector<cv::Point> data); 
	std::vector<double> getRectangleArroundShape(cv::Mat src, std::vector<cv::Point> data); 
	cv::Mat cutPlate(cv::Mat src, std::vector<cv::Point> data); 

    std::vector<std::vector<cv::Point>>  pca(cv::Mat src, int graytresh, int contourPointsTreshhMin,
        int contourPointsTreshhMax, int contourAreaTreshhMin, int contourAreaTreshhMax); 
	cv::Mat resizeImg(cv::Mat src); 

	//cv::Mat bothat(cv::Mat src, int m, int n);
	//int getVariance(std::vector<cv::Point> data);
	//std::vector<double> getFittingRectangleArroundShape(cv::Mat src, std::vector<cv::Point> data);
};

