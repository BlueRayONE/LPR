#include "MSER.h"
#include <chrono>

using namespace std::chrono;

MSER::MSER()
{
}


cv::Mat global;

std::vector<cv::Rect> MSER::run(cv::Mat imgOrig)
{
	cv::Mat img, grey, mser_p, mser_m, img_bk;
	img = imgOrig.clone();
	cv::cvtColor(img, grey, CV_BGR2GRAY);
	img_bk = img.clone();

	auto mser_pPair = this->mserFeature(grey, true);
	mser_p = mser_pPair.first;
	std::vector< cv::Rect > bboxes_p = mser_pPair.second;

	auto mser_mPair = this->mserFeature(grey, false);
	mser_m = mser_mPair.first;
	std::vector< cv::Rect > bboxes_m = mser_mPair.second;

	auto bboxes_p_pre = preDiscardBBoxes_p(bboxes_p, bboxes_m);


	cv::Mat colorP;
	cvtColor(mser_p, colorP, CV_GRAY2RGB);

	for (auto rect : bboxes_p_pre)
	{
		cv::rectangle(colorP, rect, cv::Scalar(0, 0, 255), 1);
	}

	global = colorP;

	auto bboxes_p_real = realDiscardBBoxes_p(bboxes_p_pre, bboxes_m);
	auto bboxes_p_post = postDiscardBBoxes_p(bboxes_p_real, bboxes_m); 

	//TODO: real candidate overlapped by false canidate which has more inner elements f.ex. shadows or edges
	//see R8 (should use biggest) vs MAZDA (cropped) (should use smalles)
	//keep bigger if mser+ has same size mser- maybe plus different size (it's okay to add junk to candidate if bigger has more license plate digits, keep smaller if bigger adds only more junk)
	//TODO: scale down big pictures and scale up found candidate rectangle

	std::vector<cv::Mat> canidates;

	for (auto rect : bboxes_p_real)
	{
		int w = RELAX_PIXELS;
		int x = (rect.x - w < 0) ? 0 : rect.x - w;
		int y = (rect.y - w < 0) ? 0 : rect.y - w;

		cv::rectangle(colorP, cv::Rect(x, y, rect.width + w, rect.height + w), cv::Scalar(0, 255, 255), 1);
		cv::rectangle(img, cv::Rect(x, y, rect.width + w, rect.height + w), cv::Scalar(0, 255, 255), 1);
	}

	for (auto rect : bboxes_p_post)
	{
		cv::rectangle(colorP, rect, cv::Scalar(0, 255, 0), 1);
		cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 1);
		canidates.push_back(img_bk(rect));
	}

	cv::Mat colorM;
	cvtColor(mser_m, colorM, CV_GRAY2RGB);
	for (cv::Rect rect : bboxes_m)
	{
		cv::rectangle(colorM, rect, cv::Scalar(0, 255, 255), 1);
	}


	ImageViewer::viewImage(colorP, "canidate mser_p", 400);
	ImageViewer::viewImage(mser_p, "response mser_p", 400);
	ImageViewer::viewImage(colorM, "response mser_m", 400);
	ImageViewer::viewImage(img, "candidates", 400);

	size_t i = 0;
	for (auto roi : canidates)
	{
		ImageViewer::viewImage(roi, "candidate " + std::to_string(i));
		i++;
	}

	return bboxes_p_post;
}


//returns MSER-feature points and their bounding box as pairs
std::pair< cv::Mat, std::vector<cv::Rect>> MSER::mserFeature(cv::Mat grey, bool plus)
{
	cv::Mat mser, grey2;
	if (!plus)
		cv::bitwise_not(grey, grey2);
	else
		grey2 = grey;

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

    cv::Ptr<cv::MSER> ptr = cv::MSER::create(5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5); //all default values
    ptr->setPass2Only(true);

	mser = cv::Mat(grey.rows, grey.cols, CV_8U, cv::Scalar(0));


	std::vector< std::vector< cv::Point > > msers;
	std::vector< cv::Rect > bboxes;

    ptr->detectRegions(grey2, msers, bboxes);

	for (std::vector<cv::Point> v : msers)
	{
		for (cv::Point pt : v)
		{
			mser.at<uchar>(pt) = 255;
		}
	}

	return std::make_pair(mser, bboxes);

}


std::vector<cv::Rect> MSER::preDiscardBBoxes_p(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<std::pair<cv::Rect, int>> rectInnerElements = getInnerElements(boxes_p, boxes_v);

	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };


	std::vector<std::pair<cv::Rect, int>> res;
	if (rectInnerElements.size() == 0) return std::vector<cv::Rect>();
	res.push_back(rectInnerElements[0]);

	for (size_t i = 1; i < rectInnerElements.size(); i++)
	{
		auto elem1 = rectInnerElements[i];
		bool intersect = false;
		for (size_t j = 0; j < res.size(); j++)
		{
			auto elem2 = res[j];
			if (intersectArea(elem1.first, elem2.first)) //maybe dont use overlap but real inner rectangle instead
			{

				if (std::abs(elem1.second - elem2.second) <= 2) //they have (almost) same amount of inner elements --> keep inner
				{
					intersect = true;
					res[j] = elem1;
				}

			}
		}

		if (!intersect)
			res.push_back(elem1);
	}

	std::vector<cv::Rect> res_flattened;
	for (auto elem : res)
		res_flattened.push_back(elem.first);

	return res_flattened;

}


std::vector<cv::Rect> MSER::realDiscardBBoxes_p(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<cv::Rect> res;
	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };

	cv::Mat reset = global.clone();

	for (auto rect_p : boxes_p)
	{
		std::vector<cv::Rect> innerElements;
		std::vector<cv::Point2f> centerPoints;
		global = reset.clone();
		cv::rectangle(global, rect_p, cv::Scalar(255, 0, 0), 1);
		for (auto rect_v : boxes_v)
		{
			if (intersectArea(rect_p, rect_v) == rect_v.area()) //rect_v completely in rect_p
			{
				innerElements.push_back(rect_v);
				centerPoints.push_back(cv::Point2f(rect_v.x + rect_v.width / 2, (rect_v.y + rect_v.height / 2)));
				cv::rectangle(global, rect_v, cv::Scalar(0, 255, 0), 1);
			}
		}

		if (centerPoints.size() == 0) continue; //really should not happen unless preprocess stage failed or was not called

		//STEP 1: fit line (robust)
		cv::Vec4f line;
		cv::fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01);
		float x0 = line[2];	float y0 = line[3]; float dx = line[0]; float dy = line[1];
		cv::line(global, cv::Point(x0 + dx * -1000, y0 + dy * -1000), cv::Point(x0 + dx * 1000, y0 + dy * 1000), cv::Scalar(255, 255, 0), 1);

		//STEP 1.1: check angle
		float angle = std::atan2(dy, dx);
		if (std::abs(angle) > MAX_RADIENT_ALLOWED)
			continue;


		//STEP 1.2: check number of outliers
		float a = -dy; float b = dx; float c = (dy * x0 - dx * y0); //-(-dy * x0 + dx * y0)
		auto algDist = [a, b, c](cv::Point p) { return std::abs(a*p.x + b*p.y + c); };

		std::vector<cv::Point2f> outliers;
		std::vector<cv::Rect> inlierRects;

		for (size_t i = 0; i < centerPoints.size(); i++)
		{
			cv::Point current = centerPoints[i];
			if (algDist(current) > MIN_DISTANCE_OUTLIER)
				outliers.push_back(current);
			else
				inlierRects.push_back(innerElements[i]);
		}

		if (outliers.size() > MAX_PART_OUTLIERS_ALLOWED * centerPoints.size())
			continue;


		for (auto rect : inlierRects)
		{
			cv::rectangle(global, rect, cv::Scalar(0, 255, 255), 1);
		}

	
		//war mal innerElements
		//discard outlier
		auto tuple = sameSize(inlierRects); //tuple(bool sameSize, avgHeight, avgWidth)

		//STEP 2: check if MSER- bboxes equal in size
		if (!std::get<0>(tuple)) //not same Height --> discard rect_p
			continue;

		//STEP 3: check if MSER+ bbox almost same size as average height
		if (rect_p.height - MAX_BBOX_HEIGHT_SCALE * std::get<1>(tuple) > 0)
			continue;

		//all steps passed --> push back
		res.push_back(rect_p);

	}

	global = reset.clone();
	return res;

}


//returns true if all almost same height and returns avg height, avg width
std::tuple<bool, float, float> MSER::sameSize(std::vector<cv::Rect> innerElements)
{
	auto power2 = [](double x) { return x * x; };
	uint sumWidth = 0;
	uint sumHeight = 0;

	for (auto rect : innerElements)
	{
		sumWidth += rect.width;
		sumHeight += rect.height;
	}

	double meanWidth = 1.0 * sumWidth / innerElements.size();
	double meanHeight = 1.0 * sumHeight / innerElements.size();

	double sqrSumWidth = 0; double sqrSumHeight = 0;
	for (auto rect : innerElements)
	{
		sqrSumWidth += power2(rect.width - meanWidth);
		sqrSumHeight += power2(rect.height - meanHeight);
	}

	double stdevWidth = std::sqrt(sqrSumWidth / innerElements.size());
	double stdevHeight = std::sqrt(sqrSumHeight / innerElements.size());


	int minHeight = -1; int maxHeight = -1;
	int minWidth = -1; int maxWidth = -1;
	float avgHeight = 0; float avgWidth = 0;

	size_t count = 0;
	for (auto elem : innerElements)
	{
		float leftBoundary0 = meanHeight - 2 * stdevHeight;
		float rightBoundary0 = meanHeight + 2 * stdevHeight;
		float leftBoundary1 = meanWidth - 2 * stdevWidth;
		float rightBoundary1 = meanWidth + 2 * stdevWidth;

		if (leftBoundary0 < elem.height && elem.height < rightBoundary0 && leftBoundary1 < elem.width && elem.width < rightBoundary1)
		{
			if (elem.height > maxHeight)
				maxHeight = elem.height;
			if (elem.height < minHeight || minHeight == -1)
				minHeight = elem.height;

			if (elem.width > maxWidth)
				maxWidth = elem.width;
			if (elem.width < minWidth || minWidth == -1)
				minWidth = elem.width;

			avgHeight += elem.height; avgWidth += elem.width;

			count++;
		}
	}

	avgHeight /= count; avgWidth /= count;

	bool equal = false;
	if (maxHeight <= MAX_HEIGHT_SCALE * minHeight && maxWidth <= MAX_WIDTH_SCALE * minWidth)
		equal = true;

	return std::make_tuple(equal, avgHeight, avgWidth);
	
}


std::vector<std::pair<cv::Rect, int>> MSER::getInnerElements(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<std::pair<cv::Rect, int>> rectInnerElements;

	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };

	for (auto rect_p : boxes_p)
	{
		int count = 0;
		for (auto rect_v : boxes_v)
		{
			if (intersectArea(rect_p, rect_v) == rect_v.area()) //rect_v completely in rect_p
			{
				count++;
			}
		}

		if (count > 0)
		{
			rectInnerElements.push_back(std::make_pair(rect_p, count));
		}
	}

	return rectInnerElements;

}


std::vector<cv::Rect> MSER::postDiscardBBoxes_p(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<std::pair<cv::Rect, int>> rectInnerElements = getInnerElements(boxes_p, boxes_v);

	auto intersectArea = [](cv::Rect r1, cv::Rect r2) { return (r1 & r2).area(); };

	std::vector<std::pair<cv::Rect, int>> res;
	if (rectInnerElements.size() == 0) return std::vector<cv::Rect>();
	res.push_back(rectInnerElements[0]);

	cv::Mat reset = global.clone();

	for (size_t i = 1; i < rectInnerElements.size(); i++)
	{
		auto elem1 = rectInnerElements[i];
		bool intersect = false;
		global = reset.clone();
		cv::rectangle(global, elem1.first, cv::Scalar(0, 255, 0), 1);
		for (size_t j = 0; j < res.size(); j++)
		{
			auto elem2 = res[j];
			cv::rectangle(global, elem2.first, cv::Scalar(255, 255, 0), 1);

			if (intersectArea(elem1.first, elem2.first) == elem2.first.area())
			{

				if (elem1.second == elem2.second)
				{
					intersect = true;
					res[j] = elem2;
				}
				else if (elem1.second > elem2.second)
				{
					intersect = true;
					res[j] = elem1;
				}

			}
			//swap roles
			if (intersectArea(elem1.first, elem2.first) == elem1.first.area())
			{

				if (elem1.second == elem2.second)
				{
					intersect = true;
					res[j] = elem1;
				}
				else if (elem1.second < elem2.second)
				{
					intersect = true;
					res[j] = elem2;
				}

			}
		}

		if (!intersect)
			res.push_back(elem1);
	}

	std::vector<cv::Rect> res_flattened;
	for (auto elem : res)
	{
		//check if rect is too wide
		if (elem.first.width <= elem.first.height * MAX_ASPECT_RATIO && elem.first.width > elem.first.height)
		{
			//and relax borders by RELAX_PIXELS
			int w = RELAX_PIXELS;
			int x = (elem.first.x - w < 0) ? 0 : elem.first.x - w;
			int y = (elem.first.y - w < 0) ? 0 : elem.first.y - w;
			res_flattened.push_back(cv::Rect(x, y, elem.first.width + w, elem.first.height + w));
		}
			

	}
		

	return res_flattened;

}



MSER::~MSER()
{
}
