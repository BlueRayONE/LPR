#pragma once
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "ImageViewer.h"
#include <iostream>

//#include "MainWindow.hpp"

// für SURF object detection
//#include <stdio.h>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

class licencePlateRecognition
{
public:
	/* Konstruktor */
	licencePlateRecognition();

	/* Destruktor */
	~licencePlateRecognition();



//private:
	/* Methode berechnet Sobelmaske in x Richtung und gibt den Wert zurück
	@param greyImg Grauwertbild welches bearbeitet werden soll
	@param x koordinate
	@param y koordinate */

	double _scaleFactor;

	cv::Mat computeSobelImgX(cv::Mat src);
	int xGradient(cv::Mat greyImg, int x, int y);

	cv::Mat computeRegions(cv::Mat src);
	cv::Mat closeOperation3x3(cv::Mat src);
	cv::Mat bothat(cv::Mat src, int m, int n);
	void identidy3x30(cv::Mat src, cv::Mat dst);

	void mergeResults(cv::Mat c1, cv::Mat c2, cv::Mat c3, cv::Mat dst);
	std::vector<std::vector<cv::Point>>  pca(cv::Mat src, int graytresh, int contourPointsTreshhMin, int contourPointsTreshhMax, int contourAreaTreshhMin, int contourAreaTreshhMax);
	
	int mask3x30(cv::Mat src, int x, int y);
	int sameColor(cv::Mat src, int x, int y);

	cv::Mat findPlate(cv::Mat original, cv::Mat src, std::vector<std::vector<cv::Point>> contours);
	int getBestScore(std::vector<std::vector<double>> data);
	int getVariance(std::vector<cv::Point> data);

	double getEigenVector(std::vector<cv::Point> data);
	std::vector<double> getRectangleArroundShape(cv::Mat src, std::vector<cv::Point> data);
	std::vector<double> getFittingRectangleArroundShape(cv::Mat src, std::vector<cv::Point> data);
	cv::Mat cutPlate(cv::Mat src, std::vector<cv::Point> data);

	cv::Mat pca(cv::Mat src);
	cv::Mat resizeImg(cv::Mat src);

	//void objectDetectionSURF(cv::Mat object, cv::Mat plate);
};

