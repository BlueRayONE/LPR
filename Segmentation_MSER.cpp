#include "Segmentation_MSER.h"

const float DIST_TO_MEAN = 2.0f;
const int RELAX_PIXELS_VERT = 10;
const int RELAX_PIXELS_HOR = 5;
const float MAX_ALLOWED_HEIGHT_DEV = 0.3f; //30 percent
const float MAX_ALLOWED_WIDTH_DEV = 0.75f; //40 percent

cv::Mat src, dst;

Segmentation_MSER::Segmentation_MSER()
{

}

std::vector<cv::Mat> Segmentation_MSER::findChars(cv::Mat img)
{
    cv::Mat grey;
	cv::cvtColor(img, grey, CV_BGR2GRAY);

    auto mser = MSER::mserFeature(grey, false);
    std::vector<cv::Rect> bbox = mser.second;
	cv::Mat mser_m = mser.first;
	cv::Mat resImg;
	cv::bitwise_not(mser_m, resImg);


	cv::Mat colorM;
	cvtColor(mser_m, colorM, CV_GRAY2RGB);

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 0, 255), 1);
	}

	bbox = Segmentation_MSER::discardOverlapping(bbox, grey.rows, grey.cols);

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 255, 255), 1);
	}

	bbox = Segmentation_MSER::discardOutlier(bbox);

	std::sort(bbox.begin(), bbox.end(), [](cv::Rect r1, cv::Rect r2) { return r1.x < r2.x; });

	std::vector<cv::Mat> res;

	for (auto box : bbox)
	{
		cv::Rect relaxedBox = Segmentation_MSER::relaxRect(box, grey.rows, grey.cols);
		cv::rectangle(colorM, relaxedBox, cv::Scalar(0, 255, 0), 2);
		cv::Mat digit = resImg(box);
		cv::Mat bordered;
		cv::copyMakeBorder(digit, bordered, RELAX_PIXELS_VERT, RELAX_PIXELS_VERT, RELAX_PIXELS_HOR, RELAX_PIXELS_HOR, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(255));
		res.push_back(bordered);
	}

	ImageViewer::viewImage(colorM, "mser-", 200);

	//uncomment to print passed digits
	/*size_t i = 0;
	for (auto digit : res)
	{
		ImageViewer::viewImage(digit, "digit" + std::to_string(i++), 50);
	}*/


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

	cv::Ptr<cv::MSER> ptr = cv::MSER::create(5, 10, 14400, 0.1, 0.2, 100, 1.01, 0.003, 5); //all default values
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
	int width = (x + rect.width + 2 * w_x > cols) ? 
		cols - x : 
		rect.width + 2 * w_x;
	int height = (y + rect.height + 2 * w_y > rows) ? 
		rows - y : 
		rect.height + 2 * w_y;
	return cv::Rect(x, y, width, height);

}

std::vector<cv::Rect> Segmentation_MSER::discardOverlapping(std::vector<cv::Rect> bbox, int rows, int cols)
{
	if (bbox.size() == 0) return bbox;

	auto getCenter = [](cv::Rect rect) { return cv::Point2f(rect.x + rect.width / 2, (rect.y + rect.height / 2)); };

	std::vector<cv::Point2f> centerPoints;
	for (auto rect : bbox)
	{
		centerPoints.push_back(getCenter(rect));
	}

	cv::Vec4f line;
	cv::fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01);
	float x0 = line[2];	float y0 = line[3]; float dx = line[0]; float dy = line[1];
	//cv::line(visualize_p, cv::Point(x0 + dx * -1000, y0 + dy * -1000), cv::Point(x0 + dx * 1000, y0 + dy * 1000), cv::Scalar(255, 255, 0), 1);

	float a = -dy; float b = dx; float c = (dy * x0 - dx * y0); //-(-dy * x0 + dx * y0)
	auto algDist = [a, b, c](cv::Point2f p) { return std::abs(a*p.x + b*p.y + c); };

	std::vector<cv::Rect> inlierRects;

	for (size_t i = 0; i < centerPoints.size(); i++)
	{
		cv::Point current = centerPoints[i];
		if (algDist(current) <= 1.0/DERIV * rows)
			inlierRects.push_back(bbox[i]);
	}

	//discard inner elements
	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };

	std::vector<cv::Rect> res;
	res.push_back(inlierRects[0]);

	for (size_t i = 1; i < inlierRects.size(); i++)
	{
		auto elem1 = inlierRects[i];
		bool intersect = false;
		for (size_t j = 0; j < res.size(); j++)
		{
			auto elem2 = res[j];
			if (intersectArea(elem1, elem2) == elem1.area()) //elem1 is surrounded by elem2 --> keep elem2 if center point is 
			{
				intersect = true;
				cv::Point2f center1 = getCenter(elem1);
				cv::Point2f center2 = getCenter(elem2);
				if (std::abs(center1.x - center2.x) <= 1.0 / DERIV * rows && std::abs(center1.y - center2.y) <= 1.0 / DERIV * rows)
					res[j] = elem2;
				else
					res[j] = elem1;
			}
			else if (intersectArea(elem1, elem2) == elem2.area()) //elem2 is surrounded by elem1
			{
				intersect = true;
				cv::Point2f center1 = getCenter(elem1);
				cv::Point2f center2 = getCenter(elem2);
				if (std::abs(center1.x - center2.x) <= 1.0 / DERIV * rows && std::abs(center1.y - center2.y) <= 1.0 / DERIV * rows)
					res[j] = elem1;
				else
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



