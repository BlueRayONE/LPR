#include "licencePlateRecognition.hpp"

//#define DEV

using namespace std;
using namespace cv;

licencePlateRecognition::licencePlateRecognition()
{
}

licencePlateRecognition::~licencePlateRecognition()
{
}


/**
Main function to find the plate.
Computes 3 candidate images and merges them to 1 image. Then the PCA-algirthm is applyed.
Finally the best region is chosen, cutted and returned.
@Param image
*/
Mat licencePlateRecognition::getPlate(Mat src){

	Mat smalImg = resizeImg(src);

	Mat originalImg = src.clone();
	Mat imgX = smalImg.clone();

	// c 2
	cv::Mat c2;
	c2 = computeRegions(imgX);  //same color fkt
	medianBlur(c2, c2, 5);

	// c 3
	cv::Mat c3;
	cvtColor(imgX, c3, CV_BGR2GRAY);
	c3 = closeOperation3x3(c3);


	// c 1
	cv::Mat c1;
	cvtColor(imgX, c1, CV_BGR2GRAY);
	c1 = computeSobelImgX(c1);
	medianBlur(c1, c1, 5);
	identidy3x30(c1, c1);


	//Mat resized;
	//double width = imgX.rows;
	//double x = 100/ width;
	//cv::resize(imgX, resized, Size(), x, x, 1);
	//resized = imgX.clone();
	//cv::Mat rc1;
	//cvtColor(resized, rc1, CV_BGR2GRAY);
	//rc1 = bothat(rc1, 2, 8);
	//rc1 = computeSobelImgX(rc1);
	//medianBlur(rc1, rc1, 5);
	//identidy3x30(rc1, rc1);

	//candidates
	Mat dst;
	cvtColor(imgX, dst, CV_BGR2GRAY);

	// merging all 3 candidateimages
	mergeResults(c1, c2, c3, dst);

	Mat dst2 = dst.clone();

	cvtColor(dst2, dst2, CV_GRAY2BGR);
	dilate(dst2, dst2, Mat());
	medianBlur(dst2, dst2, 5);
	dilate(dst2, dst2, Mat());

	Mat trashh;
	Mat trashhgray;
	cvtColor(dst2, trashhgray, COLOR_BGR2GRAY);
	threshold(trashhgray, trashh, 40, 255, CV_THRESH_BINARY);
	//P C A
	vector<vector<Point>> goodContours;

	goodContours = pca(dst2, 40, 150, 600, 500, 6000); //-----Values for PCA

	cv::Mat plate = Mat::zeros(1,1, src.type());

	//finding best candidate
	if (goodContours.size() > 0){
		//standart nummernschild 520mmx110mm verh.:~4.72
		plate = findPlate(originalImg, dst2, goodContours);
	}

#ifdef DEV
	cv::imshow("c3", c3);
	cv::imshow("c2", c2);
	cv::imshow("c1", c1);
    cv::imshow("dst", dst);
	cv::imshow("trashh", trashh);
	cv::imshow("trashhgray", trashhgray);

	//imwrite("Ausarbeitung/PCA/DSC03001/c3.jpg", c3);
	//imwrite("Ausarbeitung/PCA/DSC03001/c2.jpg", c2);
	//imwrite("Ausarbeitung/PCA/DSC03001/c1.jpg", c1);
	//imwrite("Ausarbeitung/PCA/DSC03001/dst.jpg", dst);
    //imwrite("Ausarbeitung/PCA/DSC03001/dst2.jpg", dst2);
	//imwrite("Ausarbeitung/PCA/DSC03001/trashh.jpg", trashh);
	//imwrite("Ausarbeitung/PCA/DSC03001/trashhgray.jpg", trashhgray);
	//imwrite("Ausarbeitung/PCA/DSC03001/plate.jpg", plate);
#endif

    cv::imshow("dst2", dst2);
	cv::imshow("plate", plate);
	return plate;
}

/**
Resizes image
*/
Mat  licencePlateRecognition::resizeImg(cv::Mat src){
	_scaleFactor = 1;
	Mat smalImg = src.clone();
	if (src.cols > 800 || src.cols < 400)
	{
		_scaleFactor = 600 / double(smalImg.cols);
		cv::resize(smalImg, smalImg, Size(), _scaleFactor, _scaleFactor, CV_INTER_AREA);
	}
	return smalImg;
}

/**
Applies mask3x30-function to all pixels of the image
@Param image
@Param output image
*/
void licencePlateRecognition::identidy3x30(cv::Mat src, cv::Mat dst){

	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 15; x < src.cols - 16; x++){
			int sum = mask3x30(src, x, y); // +yGradient(greyImg, x, y);
			dst.at<uchar>(y, x) = sum > 255 ? 255 : sum;
		}
	}
}

/**
Computes identity function with 3x30 mask
@Param image
@Param x-koordinate
@Param y-kooridante
*/
int licencePlateRecognition::mask3x30(cv::Mat src, int x, int y){

	int x0 = src.at<uchar>(y - 1, x + 15) + src.at<uchar>(y - 1, x + 14) + src.at<uchar>(y - 1, x + 13) + src.at<uchar>(y - 1, x + 12) + src.at<uchar>(y - 1, x + 11) + src.at<uchar>(y - 1, x + 10) + src.at<uchar>(y - 1, x + 9) + src.at<uchar>(y - 1, x + 8) + src.at<uchar>(y - 1, x + 7) + src.at<uchar>(y - 1, x + 6) + src.at<uchar>(y - 1, x + 5) + src.at<uchar>(y - 1, x + 4) + src.at<uchar>(y - 1, x + 3) + src.at<uchar>(y - 1, x + 2) + src.at<uchar>(y - 1, x + 1) + src.at<uchar>(y - 1, x) +
		src.at<uchar>(y - 1, x - 15) + src.at<uchar>(y - 1, x - 14) + src.at<uchar>(y - 1, x - 13) + src.at<uchar>(y - 1, x - 12) + src.at<uchar>(y - 1, x - 11) + src.at<uchar>(y - 1, x - 10) + src.at<uchar>(y - 1, x - 9) + src.at<uchar>(y - 1, x - 8) + src.at<uchar>(y - 1, x - 7) + src.at<uchar>(y - 1, x - 6) + src.at<uchar>(y - 1, x - 5) + src.at<uchar>(y - 1, x - 4) + src.at<uchar>(y - 1, x - 3) + src.at<uchar>(y - 1, x - 2) + src.at<uchar>(y - 1, x - 1);
	int x1 = src.at<uchar>(y, x + 15) + src.at<uchar>(y, x + 14) + src.at<uchar>(y, x + 13) + src.at<uchar>(y, x + 12) + src.at<uchar>(y, x + 11) + src.at<uchar>(y, x + 10) + src.at<uchar>(y, x + 9) + src.at<uchar>(y, x + 8) + src.at<uchar>(y, x + 7) + src.at<uchar>(y, x + 6) + src.at<uchar>(y, x + 5) + src.at<uchar>(y, x + 4) + src.at<uchar>(y, x + 3) + src.at<uchar>(y, x + 2) + src.at<uchar>(y, x + 1) + src.at<uchar>(y, x) +
		src.at<uchar>(y, x - 15) + src.at<uchar>(y, x - 14) + src.at<uchar>(y, x - 13) + src.at<uchar>(y, x - 12) + src.at<uchar>(y, x - 11) + src.at<uchar>(y, x - 10) + src.at<uchar>(y, x - 9) + src.at<uchar>(y, x - 8) + src.at<uchar>(y, x - 7) + src.at<uchar>(y, x - 6) + src.at<uchar>(y, x - 5) + src.at<uchar>(y, x - 4) + src.at<uchar>(y, x - 3) + src.at<uchar>(y, x - 2) + src.at<uchar>(y, x - 1);
	int x2 = src.at<uchar>(y + 1, x + 15) + src.at<uchar>(y + 1, x + 14) + src.at<uchar>(y + 1, x + 13) + src.at<uchar>(y + 1, x + 12) + src.at<uchar>(y + 1, x + 11) + src.at<uchar>(y + 1, x + 10) + src.at<uchar>(y + 1, x + 9) + src.at<uchar>(y + 1, x + 8) + src.at<uchar>(y + 1, x + 7) + src.at<uchar>(y + 1, x + 6) + src.at<uchar>(y + 1, x + 5) + src.at<uchar>(y + 1, x + 4) + src.at<uchar>(y + 1, x + 3) + src.at<uchar>(y + 1, x + 2) + src.at<uchar>(y + 1, x + 1) + src.at<uchar>(y + 1, x) +
		src.at<uchar>(y + 1, x - 15) + src.at<uchar>(y + 1, x - 14) + src.at<uchar>(y + 1, x - 13) + src.at<uchar>(y + 1, x - 12) + src.at<uchar>(y + 1, x - 11) + src.at<uchar>(y + 1, x - 10) + src.at<uchar>(y + 1, x - 9) + src.at<uchar>(y + 1, x - 8) + src.at<uchar>(y + 1, x - 7) + src.at<uchar>(y + 1, x - 6) + src.at<uchar>(y + 1, x - 5) + src.at<uchar>(y + 1, x - 4) + src.at<uchar>(y + 1, x - 3) + src.at<uchar>(y + 1, x - 2) + src.at<uchar>(y + 1, x - 1);

	int xg = (x0 + x1 + x2) / (30 * 3);
	return xg;
}


/**
Applies xGradient-function to all pixels of the image
@Param image
*/
cv::Mat licencePlateRecognition::computeSobelImgX(cv::Mat src){
	
	cv::Mat greyImg;
	//cv::cvtColor(src, greyImg, cv::COLOR_BGR2GRAY);
	if (src.channels() == 3)
	{
		cvtColor(src, greyImg, CV_BGR2GRAY);
	}
	else
	{
		greyImg = src;
	}

	cv::Mat dst = cv::Mat::zeros(src.size(), greyImg.type());
	dst.setTo(255);

	double min, max;
	cv::minMaxLoc(dst, &min, &max);

	// 0 - 255
	//for (int y = 0; y < src.rows; y++){
	//	for (int x = 0; x < src.cols; x++){
	//		int i = dst.at<uchar>(y, x);
	//		dst.at<uchar>(y, x) = 255 * ((i - min) / (max - min));
	//	}
	//}

	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 1; x < src.cols - 1; x++){
			int sum = xGradient(greyImg, x, y);
			dst.at<uchar>(y, x) = sum;
		}
	}

	return dst;
}

/** Methode berechnet Sobelmaske in x Richtung und gibt den Wert zurück
@param greyImg Grauwertbild welches bearbeitet werden soll
@param x-koordinate
@param y-koordinate */
int licencePlateRecognition::xGradient(cv::Mat greyImg, int x, int y){
	int xg = greyImg.at<uchar>(y - 1, x - 1) +
		2 * greyImg.at<uchar>(y, x - 1) +
		greyImg.at<uchar>(y + 1, x - 1) -
		greyImg.at<uchar>(y - 1, x + 1) -
		2 * greyImg.at<uchar>(y, x + 1) -
		greyImg.at<uchar>(y + 1, x + 1);
	return sqrt(xg * xg);
}

/**
Applies dilate-function and erode-function with mask of size m x n.
@Param image
@Param m number of rows
@Param n number of columns
*/
//cv::Mat licencePlateRecognition::bothat(cv::Mat src, int m, int n){
//
//	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
//				cv::Size(n ,m ),
//				cv::Point(-1, -1));
//
//	cv::Mat hlp;
//	cv::dilate(src, hlp, element);
//	cv::Mat dst;
//	cv::erode(hlp, dst, element);
//
//
//
//	for (int y = 0; y < src.rows; y++){
//		for (int x = 0; x < src.cols; x++){
//			//int a = src.at<uchar>(y, x);
//			//dst.at<uchar>(y, x) = -10;
//			//int b = dst.at<uchar>(y, x);
//			if (dst.at<uchar>(y, x) > src.at<uchar>(y, x)){
//				dst.at<uchar>(y, x) = dst.at<uchar>(y, x) - src.at<uchar>(y, x);
//			}
//			else
//			{
//				dst.at<uchar>(y, x) = src.at<uchar>(y, x) - dst.at<uchar>(y, x);
//			}
//		}
//	}
//
//
//	//// 0 - 255
//	//double min, max;
//	//cv::minMaxLoc(dst, &min, &max);
//	//for (int y = 0; y < src.rows; y++){
//	//	for (int x = 0; x < src.cols; x++){
//	//		int i = dst.at<uchar>(y, x);
//	//		dst.at<uchar>(y, x) = 255 * ((i - min) / (max - min));
//	//	}
//	//}
//
//	return dst;
//}

/**
Applies sameColor-function to all pixels of the image.
@Param image
*/
cv::Mat licencePlateRecognition::computeRegions(cv::Mat src){

	cv::Mat regions = cv::Mat::zeros(src.size(), src.type());

	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			int sum = sameColor(src, x, y);
			regions.at<cv::Vec3b>(y, x)[2] = sum;
		}
	}
	return regions;
}

/**
Finds regions with greyish color.
if all rgb-colorspaces are the same (+- smal threshold value return 255 else 0 
@Param image
@Param x-koordinate
@Param y-koordinate
*/
int licencePlateRecognition::sameColor(cv::Mat src, int x, int y){

	int b = src.at<cv::Vec3b>(y, x)[0];
	int g = src.at<cv::Vec3b>(y, x)[1];
	int r = src.at<cv::Vec3b>(y, x)[2];
	int threshold = 20;
		if (b < g + threshold && b > g - threshold && g < r + threshold && g > r - threshold){
		return 255; 
	}
	if (b == g && g == r){
		return 255;
	}
	else
	{
		return 0;
	}
}

/**
close operation with 3x3 neighbourhood
@Param image
*/
cv::Mat licencePlateRecognition::closeOperation3x3(cv::Mat src){

	cv::Mat hlp;
	cv::dilate(src, hlp, cv::Mat());
	cv::Mat dst;
	cv::erode(hlp, dst, cv::Mat());

	return dst;
}

/**
merges the 3 candidate images into one image
@Param c1 image
@Param c2 image
@Param c3 image
@Param output image
*/
void licencePlateRecognition::mergeResults(cv::Mat src, cv::Mat c2, cv::Mat c3, cv::Mat dst){
	cv::Mat candidate1 = cv::Mat::zeros(src.size(), CV_32SC1);

	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			int sum = src.at<uchar>(y, x) * (c2.at<cv::Vec3b>(y, x)[2] / 255) *c3.at<uchar>(y, x);

			if (c3.at<uchar>(y, x) > 50){
				sum = sum * c3.at<uchar>(y, x);
			}

			candidate1.at<int>(y, x) = sum;
		}
	}

	//// 0 - 255
	double min, max;
	cv::minMaxLoc(candidate1, &min, &max);
	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			int i = candidate1.at<int>(y, x);
			int sum = 255 * ((i - min) / (max - min));
			dst.at<uchar>(y, x) = sum;
		}
	}


}

/**
draws the axis
*/
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
	//! [visualization1]
	double angle;
	double hypotenuse;
	angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
	//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range

	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, CV_AA);

	// create the arrow hooks
	p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 1, CV_AA);

	p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 1, CV_AA);
	//! [visualization1]
}

/**
* @function getOrientation
*/
double getOrientation(const vector<Point> &pts, Mat &img)
{
	//! [pca]
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}

	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	//Store the center of the object
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 1; ++i) // vorher war hier bis 2 aber gab error
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));

		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	//! [pca]
	//! [visualization]
	// Draw the principal components
	circle(img, cntr, 3, Scalar(255, 0, 255), 2);
	Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
	//Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
	drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
	//drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);

	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
	//! [visualization]

	//double angle = 0;
	return angle;
}

/**
Function to find the main components of the image using the PCA-algorithm
@Param src Image
@Param graytrash
@Param contourPointstrashhMin
@Param contourPointstrashhMax
@Param contourAreatrashhMin
@Param contourAreatrashhMax
*/
vector<vector<Point>>  licencePlateRecognition::pca(cv::Mat src, int graytrash, int contourPointstrashhMin,
	int contourPointstrashhMax, int contourAreatrashhMin, int contourAreatrashhMax){
	// Convert image to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// Convert image to binary
	Mat bw;
	threshold(gray, bw, graytrash, 255, CV_THRESH_BINARY);// | CV_THRESH_OTSU);
	//! [pre-process]

	//! [contours]
	// Find all the contours in the thresholded image
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(bw, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	vector<vector<Point> > goodContours;

	for (size_t i = 0; i < contours.size(); ++i)
	{
		// Ignore contours that have too many/less contour points
		if (contours[i].size() < contourPointstrashhMin || contourPointstrashhMax < contours[i].size()) continue;

		// Calculate the area of each contour
		double area = contourArea(contours[i]);
		// Ignore contours that Areas are too small or too large
		if (area < contourAreatrashhMin || contourAreatrashhMax < area) continue;

		//goodContours.push_back;

		vector<Point> tmp;

		for (size_t numPoints = 0; numPoints < contours[i].size(); numPoints++)
		{
			tmp.push_back(contours[i][numPoints]);
		}
		goodContours.push_back(tmp);

		// Draw each contour only for visualisation purposes
		drawContours(src, contours, static_cast<int>(i), Scalar(0, 0, 255), 2, 8, hierarchy, 0);
		// Find the orientation of each shape
		getOrientation(contours[i], src);
	}
	//! [contours]

	//imshow("output", src);

	return goodContours;
}

//regionen finden
//winkel berechnen
//drehen
//gleiche regionen wieder finden?!
//bestes finden |__|
//ausschneiden

/** finds the region with the best scoreing, cuts it and returns cutted area 
@Param original image
@Param grey image
@Param vector with points of shape */
Mat licencePlateRecognition::findPlate(cv::Mat original, Mat src, vector<vector<Point>> contours){

	vector<vector<double>> scoreing;

	for (int i = 0; i < contours.size(); i++){
		vector<double> tmp;
		//getScore(contours[i]);

		tmp.push_back(getEigenVector(contours[i]));
		vector<double> rect = getRectangleArroundShape(src, contours[i]);
		//getFittingRectangleArroundShape(src, contours[i]);//---------------------------
		tmp.push_back(rect[0]);
		tmp.push_back(rect[1]);

		scoreing.push_back(tmp);
	}
	
	int best = getBestScore(scoreing);

	Mat cutted = cutPlate(original, contours[best]);

	//--------------for warping-------------------------------
	//cv::Point2f pc(cutted.cols / 2., cutted.rows / 2.);
	//double angle = (360 / (2 * 3.14159265)) * scoreing[best].at(0);
	//cv::Mat r = cv::getRotationMatrix2D(pc, angle, 1.0);

	//cv::warpAffine(cutted, cutted, r, cutted.size());
	//---------------------------------------------------------
	return cutted;
}

//int licencePlateRecognition::getVariance(vector<Point> data){
//
//	int sumX = 0;
//	int sumY = 0;
//	for (int i = 0; i < data.size(); i++){
//		sumX += data[i].x; int test = data[i].x;
//		sumY += data[i].y;
//	}
//	double meanX = sumX / data.size();
//	double meanY = sumY / data.size();
//
//	double x = 0;
//	double y = 0;
//	for (int i = 0; i < data.size(); i++){
//		x += abs(data[i].x - meanX);
//		y += abs(data[i].y - meanY);
//	}
//	double varianceX = x / data.size();
//	double varianceY = y / data.size();
//
//	//standart nummernschild 520mmx110mm ~4.72
//	double ratio = x / y;
//
//	return 0;
//}

/** returns pos with best score 
@Param vector with scores */
int licencePlateRecognition::getBestScore(std::vector<std::vector<double>> data){
	

	double maxScore = 0;
	int pos;

	for (int i = 0; i < data.size(); i++){
		double min0 = data[i].at(0); //winkel
		double max1 = data[i].at(1); //seitenverhäldnis (4.7 perfekt)
		double max2 = data[i].at(2); //abdeckung zum Rechteck 0-1 da %
		
		double score = max2;
		if (max1 > 2.0 & max1 < 5.7){
			score = max1 * max2;
		}
		
		if (maxScore < score){
			maxScore = score;
			pos = i;
		}
	}

	return pos;
}

/** Draws rectangle arround the shape and returns its scores 
@Param image
@Param vector with points of shape */
vector<double> licencePlateRecognition::getRectangleArroundShape(Mat src, vector<Point> data){


	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
	for (int i = 0; i < data.size(); i++){
		if (data[i].x > maxX) maxX = data[i].x;
		if (data[i].y > maxY) maxY = data[i].y;
		if (data[i].x < minX) minX = data[i].x;
		if (data[i].y < minY) minY = data[i].y;
	}

    line(src, cv::Point(minX, minY), Point(minX, maxY), Scalar(255, 0, 0), 2, CV_AA); //oben -
    line(src, cv::Point(minX, maxY), Point(maxX, maxY), Scalar(255, 0, 0), 2, CV_AA); //hinten |
    line(src, cv::Point(maxX, minY), Point(maxX, maxY), Scalar(255, 0, 0), 2, CV_AA); //unten _
    line(src, cv::Point(maxX, minY), Point(minX, minY), Scalar(255, 0, 0), 2, CV_AA); //vorne |

	// Calculate the area of each contour
	vector<double> scores;
	double areaPlate = contourArea(data);
	double areaRectX = maxX - minX;
	double areaRectY = maxY - minY;
	double sideRatio = areaRectX / areaRectY;
	double areaRect = areaRectX * areaRectY;
	double areaOutsideOfPlate = areaRect - areaPlate;
	double areaRatio = areaPlate / areaRect;

	//vector<Point> contour;
	//contour.push_back(Point(minX, minY));
	//contour.push_back(Point(minX, maxY));
	//contour.push_back(Point(maxX, maxY));
	//contour.push_back(Point(maxX, minY));

	//double area = contourArea(contour);

	scores.push_back(sideRatio);
	scores.push_back(areaRatio);

	return scores;
}

//´machtfast passendes rechteck um plate
//vector<double> licencePlateRecognition::getFittingRectangleArroundShape(Mat src, vector<Point> data){
//
//	cv::Point minX = Point(INT_MAX, 0), minY = Point(0, INT_MAX), maxX = Point(INT_MIN, 0), maxY = Point(0, INT_MIN);
//	for (int i = 0; i < data.size(); i++){
//		if (data[i].x > maxX.x) maxX = data[i];
//		if (data[i].y > maxY.y) maxY = data[i];
//		if (data[i].x < minX.x) minX = data[i];
//		if (data[i].y < minY.y) minY = data[i];
//	}
//
//	line(src, minX, minY, Scalar(255, 255, 0), 1, CV_AA); //oben -  
//	line(src, minY, maxX, Scalar(255, 255, 0), 1, CV_AA); //hinten |
//	line(src, maxY, maxX, Scalar(255, 255, 0), 1, CV_AA); //unten _ 
//	line(src, minX, maxY, Scalar(255, 255, 0), 1, CV_AA); //vorne | 
//
//	vector<double> scores;
//	double areaPlate = contourArea(data);
//
//
//	vector<Point> contour;
//	contour.push_back(minX); //a
//	contour.push_back(maxY); //c
//	contour.push_back(maxX); //d
//	contour.push_back(minY); //b
//
//	double area = contourArea(contour);
//
//	double areaOutsideOfPlate = abs(area - areaPlate);
//
//	return scores;
//}

/** Computes eigenvector of given shape 
@Param vector with points of shape*/
double licencePlateRecognition::getEigenVector(vector<Point> data){
	//! [pca]
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(data.size());
	Mat data_pts = Mat(sz, 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = data[i].x;
		data_pts.at<double>(i, 1) = data[i].y;
	}

	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	//Store the center of the object
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 1; ++i) 
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));

		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

/** cuts the liceneplate out of the image 
@Param image
@Param vector with coordinates of licenceplate */
Mat licencePlateRecognition::cutPlate(Mat src, vector<Point> data){

	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
	for (int i = 0; i < data.size(); i++){
		if (data[i].x > maxX) maxX = data[i].x;
		if (data[i].y > maxY) maxY = data[i].y;
		if (data[i].x < minX) minX = data[i].x;
		if (data[i].y < minY) minY = data[i].y;
	}

	int areaRectX = maxX - minX;
	int areaRectY = maxY - minY;

	int threshhold = src.rows/200;

	int a = (minX / _scaleFactor) - threshhold;
	int b = (minY / _scaleFactor) - threshhold;
	int c = (areaRectX / _scaleFactor) + threshhold * 2;
	int d = (areaRectY / _scaleFactor) + threshhold * 2;

	if ((minX / _scaleFactor) - threshhold < 0){ a = 0; }
	if ((minY / _scaleFactor) - threshhold < 0){ b = 0; }
	if ((areaRectX / _scaleFactor) + threshhold * 2 > src.cols){ c = src.cols; }
	if ((areaRectY / _scaleFactor) + threshhold * 2 > src.rows){ d = src.rows; }

	Mat croppedImage = src(Rect(a, b, c, d));

	return croppedImage;
}
