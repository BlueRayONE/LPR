#include "Segmentation_MSER.h"

const float DIST_TO_MEAN = 2.0f;
const int RELAX_PIXELS = 2;

cv::Mat src, dst;

int morph_elem = 0;
int morph_size = 0;
int x = 0;
int y = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 100;

char* window_name = "Morphology Transformations Demo";

void Morphology_Operations(int, void*)
{
	// Since MORPH_X : 2,3,4,5 and 6
	int operation = morph_operator + 2;

	cv::Mat element = cv::getStructuringElement(morph_elem, cv::Size(src.cols / (x + 1), src.rows / (y + 1)));

	/// Apply the specified morphology operation
	morphologyEx(src, dst, operation, element);
	imshow(window_name, dst);
}

Segmentation_MSER::Segmentation_MSER()
{

}

std::vector<cv::Mat> Segmentation_MSER::findChars(cv::Mat img)
{
    cv::Mat grey;
	cv::cvtColor(img, grey, CV_BGR2GRAY);

	/*src = grey;
	/// Create window
	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to select Morphology operation
	cv::createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations);

	/// Create Trackbar to select kernel type
	cv::createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
		&morph_elem, max_elem,
		Morphology_Operations);

	/// Create Trackbar to choose kernel size
	cv::createTrackbar("Kernel size x:\n 2n +1", window_name,
		&x, max_kernel_size,
		Morphology_Operations);

	cv::createTrackbar("Kernel size:\n 2n +1", window_name,
		&y, max_kernel_size,
		Morphology_Operations);

	/// Default start
	Morphology_Operations(0, 0);*/

	grey = morph(grey);

	ImageViewer::viewImage(grey, "morphed", 400);

    auto mser = MSER::mserFeature(grey, false);
    std::vector<cv::Rect> bbox = mser.second;
	cv::Mat mser_m = mser.first;

	cv::Mat colorM;
	cvtColor(mser_m, colorM, CV_GRAY2RGB);

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 0, 255), 1);
	}

	bbox = Segmentation_MSER::discardOverlapping(bbox);

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 255, 255), 1);
	}

	bbox = Segmentation_MSER::discardOutlier(bbox);

	std::sort(bbox.begin(), bbox.end(), [](cv::Rect r1, cv::Rect r2) { return r1.x < r2.x; });

	std::vector<cv::Mat> res;
	cv::Mat img_bk = img.clone();

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 255, 0), 2);
		cv::Mat mat = img_bk(relaxedBox);
		res.push_back(mat);
	}

	ImageViewer::viewImage(colorM, "mser-", 200);


    return res;
}

cv::Rect Segmentation_MSER::relaxRect(cv::Rect rect, int rows, int cols)
{
	int w = RELAX_PIXELS;
	int x = (rect.x - w < 0) ? 0 : rect.x - w;
	int y = (rect.y - w < 0) ? 0 : rect.y - w;
	int width = (rect.x + rect.width + 2 * w > cols) ? cols - rect.x : rect.width + 2 * w;
	int height = (rect.y + rect.height + 2 * w > rows) ? rows - rect.y : rect.height + 2 * w;
	return cv::Rect(x, y, width, height);

}

std::vector<cv::Rect> Segmentation_MSER::discardOverlapping(std::vector<cv::Rect> bbox)
{
	//discard inner elements
	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };

	std::vector<cv::Rect> res;
	if (bbox.size() == 0) return bbox;
	res.push_back(bbox[0]);

	for (size_t i = 1; i < bbox.size(); i++)
	{
		auto elem1 = bbox[i];
		bool intersect = false;
		for (size_t j = 0; j < res.size(); j++)
		{
			auto elem2 = res[j];
			if (intersectArea(elem1, elem2) == elem1.area())
			{
				intersect = true;
				res[j] = elem2;
			}
			else if (intersectArea(elem1, elem2) == elem2.area())
			{
				intersect = true;
				res[j] = elem1;
			}
		}

		if (!intersect)
			res.push_back(elem1);
	}

	return res;
}

std::vector<cv::Rect> Segmentation_MSER::discardOutlier(std::vector<cv::Rect> bbox)
{
	std::vector<cv::Rect> res;

	auto tuple = MSER::meanStdDev(bbox);
	double meanWidth = std::get<0>(tuple);
	double meanHeight = std::get<1>(tuple);
	double stdevWidth = std::get<2>(tuple);
	double stdevHeight = std::get<3>(tuple);

	for (auto box : bbox)
	{
		float leftBoundary0 = meanHeight - DIST_TO_MEAN * stdevHeight;
		float rightBoundary0 = meanHeight + DIST_TO_MEAN * stdevHeight;
		float leftBoundary1 = meanWidth - DIST_TO_MEAN* stdevWidth;
		float rightBoundary1 = meanWidth + DIST_TO_MEAN * stdevWidth;

		if (leftBoundary0 < box.height && box.height < rightBoundary0 && leftBoundary1 < box.width && box.width < rightBoundary1)
		{
			res.push_back(box);
		}
	}

	return res;
}

cv::Mat Segmentation_MSER::morph(cv::Mat img)
{
	cv::Mat morphed;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(img.cols / 100, img.rows / 60));

	cv::morphologyEx(img, morphed, cv::MORPH_OPEN, element);

	return morphed;
}



