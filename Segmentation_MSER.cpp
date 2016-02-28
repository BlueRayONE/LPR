#include "Segmentation_MSER.h"

const float DIST_TO_MEAN = 2.0f;
const int RELAX_PIXELS_VERT = 10;
const int RELAX_PIXELS_HOR = 5;
const float MAX_ALLOWED_HEIGHT_DEV = 0.3f; //30 percent
const float MAX_ALLOWED_WIDTH_DEV = 0.75f; //40 percent

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

	//grey = morph(grey);

	//ImageViewer::viewImage(grey, "morphed", 400);

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

std::pair< cv::Mat, std::vector<cv::Rect>> Segmentation_MSER::mserFeature(cv::Mat grey)
{

	//int _delta=5						//how many different gray levels does a region need to be stable to be considered max stable
	//int _min_area=60					//reject too small
	//int _max_area=14400				//or too big areas		--> could be small in order to get rid of too big candidates
	//double _max_variation=0.25		//or too big variance	--> should stay small to get many bbox for same object --> robust line fit

	//next args only apply for color image
	//double _min_diversity=.2			//or too similar
	//int _max_evolution=200			//evolution steps
	//double _area_threshold=1.01		//threshold to cause re-init
	//double _min_margin=0.003			//ignore too small margin
	//int _edge_blur_size=5				//blur kernel size
	//cv::bitwise_not(grey, grey);

	cv::Ptr<cv::MSER> ptr = cv::MSER::create(5, 60, 14400, 0.25, 0.2, 100, 1.01, 0.003, 5); //all default values
	ptr->setPass2Only(true);

	cv::Mat mser = cv::Mat(grey.rows, grey.cols, CV_8U, cv::Scalar(0));


	std::vector< std::vector< cv::Point > > msers;
	std::vector< cv::Rect > bboxes;

	ptr->detectRegions(grey, msers, bboxes);

	for (std::vector<cv::Point> v : msers)
	{
		for (cv::Point pt : v)
		{
			mser.at<uchar>(pt) = 255;
		}
	}

	return std::make_pair(mser, bboxes);

}

cv::Rect Segmentation_MSER::relaxRect(cv::Rect rect, int rows, int cols)
{
	int w_x = RELAX_PIXELS_HOR;
	int w_y = RELAX_PIXELS_VERT;
	int x = (rect.x - w_x < 0) ? 0 : rect.x - w_x;
	int y = (rect.y - w_y < 0) ? 0 : rect.y - w_y;
	int width = (rect.x + rect.width + 2 * w_x > cols) ? cols - rect.x : rect.width + 2 * w_x;
	int height = (rect.y + rect.height + 2 * w_y > rows) ? rows - rect.y : rect.height + 2 * w_y;
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
				res[j] = elem1;
			}
			else if (intersectArea(elem1, elem2) == elem2.area())
			{
				intersect = true;
				res[j] = elem2;
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

	std::sort(bbox.begin(), bbox.end(), [](cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; });
	cv::Rect medianHeight = bbox[bbox.size() / 2];
	std::sort(bbox.begin(), bbox.end(), [](cv::Rect r1, cv::Rect r2) { return r1.width < r2.width; });
	cv::Rect medianWidth = bbox[bbox.size() / 2];


	double rightBoundary0 = medianHeight.height + medianHeight.height * MAX_ALLOWED_HEIGHT_DEV;
	double leftBoundary0 = medianHeight.height - medianHeight.height * MAX_ALLOWED_HEIGHT_DEV;
	double rightBoundary1 = medianWidth.width + medianWidth.width * MAX_ALLOWED_WIDTH_DEV;
	double leftBoundary1 = medianWidth.width - medianWidth.width * MAX_ALLOWED_WIDTH_DEV;
	for (auto box : bbox)
	{
		if (box.height >= leftBoundary0 && box.height <= rightBoundary0 &&
			box.width >= leftBoundary1 && box.width <= rightBoundary1)
		res.push_back(box);
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



