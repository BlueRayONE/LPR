#include "licencePlateRecognition.hpp"


using namespace std;
using namespace cv;

licencePlateRecognition::licencePlateRecognition()
{
}

licencePlateRecognition::~licencePlateRecognition()
{
}



Mat licencePlateRecognition::letTheMagicHappen(Mat src){

	Mat smalImg = resizeImg(src);


	Mat originalImg = src.clone();//smalImg.clone();
	Mat imgX = smalImg.clone();

	// h scale to 100
	// bothat operator --------
	// h sobel xgrad
	// scaling zwischen min max 0 - 255
	// smooth edges 
	// h 5x5 median 
	// convolve it with a3�30 identity mask

	// c 2
	cv::Mat c2;
	c2 = computeRegions(imgX);
	medianBlur(c2, c2, 5);

	// c 3
	cv::Mat c3;
	cvtColor(imgX, c3, CV_BGR2GRAY);
	c3 = closeOperation3x3(c3);


	// c 1


	//cv::Sobel(imgX, gradX, 0, 1, 0);

	cv::Mat c1;
	cvtColor(imgX, c1, CV_BGR2GRAY);

	//cv::morphologyEx(c1, c1, 0, );

	//c1 = closeOperation3x3(c1);
	//c1 = bothat(c1, 2, 8);
	//dilate(c1, c1, Mat());
	//erode(c1, c1, Mat());
	c1 = computeSobelImgX(c1);
	medianBlur(c1, c1, 5);
	identidy3x30(c1, c1);


	Mat resized;
	//double width = imgX.rows;
	//double x = 100/ width;
	//cv::resize(imgX, resized, Size(), x, x, 1);
	resized = imgX.clone();
	cv::Mat rc1;
	cvtColor(resized, rc1, CV_BGR2GRAY);
	//rc1 = bothat(rc1, 2, 8);
	rc1 = computeSobelImgX(rc1);
	//medianBlur(rc1, rc1, 5);
	//identidy3x30(rc1, rc1);

	//candidates
	Mat dst;
	cvtColor(imgX, dst, CV_BGR2GRAY);

	//	cv::Mat energyMap = cv::Mat::zeros(sobelImg.size(), CV_32SC1);

	mergeResults(c1, c2, c3, dst);
	//identidy3x30(dst, dst);

	Mat threshh;
	threshold(dst, threshh, 50, 255, THRESH_BINARY);
	Mat dst2 = dst.clone();

	cvtColor(dst2, dst2, CV_GRAY2BGR);
	dilate(dst2, dst2, Mat());
	medianBlur(dst2, dst2, 5);
	dilate(dst2, dst2, Mat());

	//P C A
	vector<vector<Point>> goodContours;

	goodContours = pca(dst2, 50, 150, 400, 500, 6000);
	//	 (dst2, 50, 100, 400, 100 besser 150, 6000); gute werte!!

	cv::Mat plate;

	if (goodContours.size() > 0){
		//standart nummernschild 520mmx110mm verh.:~4.72
		plate = findPlate(originalImg, dst2, goodContours);
	}
	//cv::Rect als ausgabe



	cv::imshow("c3", c3);
	cv::imshow("c2", c2);
	cv::imshow("c1", c1);
	cv::imshow("dst", dst);
	cv::imshow("dst2", dst2);
	cv::imshow("threshh", threshh);
	cv::imshow("plate", plate);

	return plate;
}


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

void licencePlateRecognition::identidy3x30(cv::Mat src, cv::Mat dst){

	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 15; x < src.cols - 16; x++){
			int sum = mask3x30(src, x, y); // +yGradient(greyImg, x, y);
			dst.at<uchar>(y, x) = sum > 255 ? 255 : sum;
		}
	}
}

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

int licencePlateRecognition::xGradient(cv::Mat greyImg, int x, int y){
	int xg = greyImg.at<uchar>(y - 1, x - 1) +
		2 * greyImg.at<uchar>(y, x - 1) +
		greyImg.at<uchar>(y + 1, x - 1) -
		greyImg.at<uchar>(y - 1, x + 1) -
		2 * greyImg.at<uchar>(y, x + 1) -
		greyImg.at<uchar>(y + 1, x + 1);
	return sqrt(xg * xg);
}


cv::Mat licencePlateRecognition::bothat(cv::Mat src, int m, int n){

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
				cv::Size(n ,m ),
				cv::Point(-1, -1));

	cv::Mat hlp;
	cv::dilate(src, hlp, element);
	cv::Mat dst;
	cv::erode(hlp, dst, element);



	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			//int a = src.at<uchar>(y, x);
			//dst.at<uchar>(y, x) = -10;
			//int b = dst.at<uchar>(y, x);
			if (dst.at<uchar>(y, x) > src.at<uchar>(y, x)){
				dst.at<uchar>(y, x) = dst.at<uchar>(y, x) - src.at<uchar>(y, x);
			}
			else
			{
				dst.at<uchar>(y, x) = src.at<uchar>(y, x) - dst.at<uchar>(y, x);
			}
		}
	}


	//// 0 - 255
	//double min, max;
	//cv::minMaxLoc(dst, &min, &max);
	//for (int y = 0; y < src.rows; y++){
	//	for (int x = 0; x < src.cols; x++){
	//		int i = dst.at<uchar>(y, x);
	//		dst.at<uchar>(y, x) = 255 * ((i - min) / (max - min));
	//	}
	//}

	return dst;
}

cv::Mat licencePlateRecognition::computeRegions(cv::Mat src){

	cv::Mat regions = cv::Mat::zeros(src.size(), src.type());//CV_32SC1);

	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			int sum = sameColor(src, x, y);
			regions.at<cv::Vec3b>(y, x)[2] = sum;
			//regions.at<int>(y, x) = sum;
		}
	}
	return regions;
}

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

cv::Mat licencePlateRecognition::closeOperation3x3(cv::Mat src){

	cv::Mat hlp;
	cv::dilate(src, hlp, cv::Mat());
	cv::Mat dst;
	cv::erode(hlp, dst, cv::Mat());

	return dst;
}


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

vector<vector<Point>>  licencePlateRecognition::pca(cv::Mat src, int graytresh, int contourPointsTreshhMin,
	int contourPointsTreshhMax, int contourAreaTreshhMin, int contourAreaTreshhMax){
	// Convert image to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// Convert image to binary
	Mat bw;
	threshold(gray, bw, graytresh, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
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
		if (contours[i].size() < contourPointsTreshhMin || contourPointsTreshhMax < contours[i].size()) continue;

		// Calculate the area of each contour
		double area = contourArea(contours[i]);
		// Ignore contours that Areas are too small or too large
		if (area < contourAreaTreshhMin || contourAreaTreshhMax < area) continue;

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


Mat licencePlateRecognition::findPlate(cv::Mat original, Mat src, vector<vector<Point>> contours){

	vector<vector<double>> scoreing;

	for (int i = 0; i < contours.size(); i++){
		vector<double> tmp;
		//getScore(contours[i]);

		tmp.push_back(getEigenVector(contours[i])); // brauch ich nicht
		vector<double> rect = getRectangleArroundShape(src, contours[i]);
		//getFittingRectangleArroundShape(src, contours[i]);//---------------------------
		tmp.push_back(rect[0]);
		tmp.push_back(rect[1]);

		scoreing.push_back(tmp);
	}
	
	int best = getBestScore(scoreing);

	Mat cutted = cutPlate(original, contours[best]);
	//imshow("cutted", cutted);//---------------------------------------------

	//cv::Point2f pc(cutted.cols / 2., cutted.rows / 2.);
	//double angle = (360 / (2 * 3.14159265)) * scoreing[best].at(0);
	//cv::Mat r = cv::getRotationMatrix2D(pc, angle, 1.0);

	//cv::warpAffine(cutted, cutted, r, cutted.size());

	//imshow("cutted warped", cutted); //---------------------------------------------
	return cutted;
}

int licencePlateRecognition::getVariance(vector<Point> data){

	int sumX = 0;
	int sumY = 0;
	for (int i = 0; i < data.size(); i++){
		sumX += data[i].x; int test = data[i].x;
		sumY += data[i].y;
	}
	double meanX = sumX / data.size();
	double meanY = sumY / data.size();

	double x = 0;
	double y = 0;
	for (int i = 0; i < data.size(); i++){
		x += abs(data[i].x - meanX);
		y += abs(data[i].y - meanY);
	}
	double varianceX = x / data.size();
	double varianceY = y / data.size();

	//standart nummernschild 520mmx110mm ~4.72
	double ratio = x / y;

	return 0;
}

int licencePlateRecognition::getBestScore(std::vector<std::vector<double>> data){
	

	double maxScore = 0;
	int pos;

	for (int i = 0; i < data.size(); i++){
		double min0 = data[i].at(0); //winkel //brauchich nicht muss  aber dann zahlen anspassen bei nachfolgenden 2
		double max1 = data[i].at(1); //seitenverh�ldnis (4.7 perfekt)
		double max2 = data[i].at(2); //abdeckung zum Rechteck 0-1 da %
		
		double score = max1 * max2;
		
		if (maxScore < score){
			maxScore = score;
			pos = i;
		}
	}

	return pos;
}

vector<double> licencePlateRecognition::getRectangleArroundShape(Mat src, vector<Point> data){


	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
	for (int i = 0; i < data.size(); i++){
		if (data[i].x > maxX) maxX = data[i].x;
		if (data[i].y > maxY) maxY = data[i].y;
		if (data[i].x < minX) minX = data[i].x;
		if (data[i].y < minY) minY = data[i].y;
	}

	line(src, cv::Point(minX, minY), Point(minX, maxY), Scalar(255, 0, 0), 1, CV_AA); //oben -
	line(src, cv::Point(minX, maxY), Point(maxX, maxY), Scalar(255, 0, 0), 1, CV_AA); //hinten |
	line(src, cv::Point(maxX, minY), Point(maxX, maxY), Scalar(255, 0, 0), 1, CV_AA); //unten _
	line(src, cv::Point(maxX, minY), Point(minX, minY), Scalar(255, 0, 0), 1, CV_AA); //vorne |

	// Calculate the area of each contour
	vector<double> scores;
	double areaPlate = contourArea(data);
	double areaRectX = maxX - minX;
	double areaRectY = maxY - minY;
	double sideRatio = areaRectX / areaRectY;
	double areaRect = areaRectX * areaRectY;
	double areaOutsideOfPlate = areaRect - areaPlate;
	double areaRatio = areaPlate / areaRect;

	vector<Point> contour;
	contour.push_back(Point(minX, minY)); //a
	contour.push_back(Point(minX, maxY)); //c
	contour.push_back(Point(maxX, maxY)); //d
	contour.push_back(Point(maxX, minY)); //b

	double area = contourArea(contour);

	scores.push_back(sideRatio);
	scores.push_back(areaRatio);

	return scores;
}

//�machtfast passendes rechteck um plate
vector<double> licencePlateRecognition::getFittingRectangleArroundShape(Mat src, vector<Point> data){

	cv::Point minX = Point(INT_MAX, 0), minY = Point(0, INT_MAX), maxX = Point(INT_MIN, 0), maxY = Point(0, INT_MIN);
	for (int i = 0; i < data.size(); i++){
		if (data[i].x > maxX.x) maxX = data[i];
		if (data[i].y > maxY.y) maxY = data[i];
		if (data[i].x < minX.x) minX = data[i];
		if (data[i].y < minY.y) minY = data[i];
	}

	line(src, minX, minY, Scalar(255, 255, 0), 1, CV_AA); //oben -     a, b
	line(src, minY, maxX, Scalar(255, 255, 0), 1, CV_AA); //hinten |   b, d
	line(src, maxY, maxX, Scalar(255, 255, 0), 1, CV_AA); //unten _    c, d
	line(src, minX, maxY, Scalar(255, 255, 0), 1, CV_AA); //vorne |    a, c

	// Calculate the area of each contour
	vector<double> scores;
	double areaPlate = contourArea(data);
	//double areaRectX = maxX - minX;
	//double areaRectY = maxY - minY;
	//double sideRatio = areaRectX / areaRectY;
	//double areaRect = areaRectX * areaRectY;
	//double areaOutsideOfPlate = areaRect - areaPlate;
	//double areaRatio = areaPlate / areaRect;


	vector<Point> contour;
	contour.push_back(minX); //a
	contour.push_back(maxY); //c
	contour.push_back(maxX); //d
	contour.push_back(minY); //b

	double area = contourArea(contour);

	double areaOutsideOfPlate = abs(area - areaPlate);

	//scores.push_back(sideRatio);
	//scores.push_back(areaRatio);

	return scores;
}

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
	for (int i = 0; i < 1; ++i) // vorher war hier bis 2 aber gab error
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));

		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	//double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);

	return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

Mat licencePlateRecognition::cutPlate(Mat src, vector<Point> data){

	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
	for (int i = 0; i < data.size(); i++){
		if (data[i].x > maxX) maxX = data[i].x;
		if (data[i].y > maxY) maxY = data[i].y;
		if (data[i].x < minX) minX = data[i].x;
		if (data[i].y < minY) minY = data[i].y;
	}

	//line(src, cv::Point(minX, minY), Point(minX, maxY), Scalar(255, 0, 0), 1, CV_AA); //oben -
	//line(src, cv::Point(minX, maxY), Point(maxX, maxY), Scalar(255, 0, 0), 1, CV_AA); //hinten |
	//line(src, cv::Point(maxX, minY), Point(maxX, maxY), Scalar(255, 0, 0), 1, CV_AA); //unten _
	//line(src, cv::Point(maxX, minY), Point(minX, minY), Scalar(255, 0, 0), 1, CV_AA); //vorne |

	int areaRectX = maxX - minX;
	int areaRectY = maxY - minY;

	int threshhold = src.rows/200;

	Mat croppedImage = src(Rect((minX / _scaleFactor) - threshhold, (minY / _scaleFactor) - threshhold, (areaRectX / _scaleFactor) + threshhold * 2, (areaRectY / _scaleFactor) + threshhold * 2));

	return croppedImage;
}

//void licencePlateRecognition::objectDetectionSURF(cv::Mat object, cv::Mat plate){
//
//	Mat img_object = object; //imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//	Mat img_scene = plate; // imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
//
//	if (!img_object.data || !img_scene.data)
//	{
//		std::cout << " --(!) Error reading images " << std::endl; //return -1;
//	}
//
//	//-- Step 1: Detect the keypoints using SURF Detector
//	int minHessian = 400;
//
//	SurfFeatureDetector detector(minHessian);
//
//	std::vector<KeyPoint> keypoints_object, keypoints_scene;
//
//	detector.detect(img_object, keypoints_object);
//	detector.detect(img_scene, keypoints_scene);
//
//	//-- Step 2: Calculate descriptors (feature vectors)
//	SurfDescriptorExtractor extractor;
//
//	Mat descriptors_object, descriptors_scene;
//
//	extractor.compute(img_object, keypoints_object, descriptors_object);
//	extractor.compute(img_scene, keypoints_scene, descriptors_scene);
//
//	//-- Step 3: Matching descriptor vectors using FLANN matcher
//	FlannBasedMatcher matcher;
//	std::vector< DMatch > matches;
//	matcher.match(descriptors_object, descriptors_scene, matches);
//
//	double max_dist = 0; double min_dist = 100;
//
//	//-- Quick calculation of max and min distances between keypoints
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//
//	printf("-- Max dist : %f \n", max_dist);
//	printf("-- Min dist : %f \n", min_dist);
//
//	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//	std::vector< DMatch > good_matches;
//
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		if (matches[i].distance < 3 * min_dist)
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	Mat img_matches;
//	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
//		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	//-- Localize the object
//	std::vector<Point2f> obj;
//	std::vector<Point2f> scene;
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		//-- Get the keypoints from the good matches
//		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
//		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
//	}
//
//	Mat H = findHomography(obj, scene, CV_RANSAC);
//
//	//-- Get the corners from the image_1 ( the object to be "detected" )
//	std::vector<Point2f> obj_corners(4);
//	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
//	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
//	std::vector<Point2f> scene_corners(4);
//
//	perspectiveTransform(obj_corners, scene_corners, H);
//
//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//
//	//-- Show detected matches
//	imshow("Good Matches & Object detection", img_matches);
//
//	//waitKey(0);
//}