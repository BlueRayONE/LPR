#include "MSER.h"
#include <chrono>

using namespace std::chrono;

MSER::MSER(cv::Mat imgOrig)
{
	originalImage = imgOrig.clone();
	resizedImage = resizeImg(originalImage);
}

std::vector<cv::Mat> MSER::run()
{
	img_bk = resizedImage.clone();
	cv::cvtColor(resizedImage, grey, CV_BGR2GRAY);

	auto mser_pPair = this->mserFeature(grey, true);
	mser_p = mser_pPair.first;
	std::vector< cv::Rect > bboxes_p = mser_pPair.second;

	auto mser_mPair = this->mserFeature(grey, false);
	mser_m = mser_mPair.first;
	std::vector< cv::Rect > bboxes_m = mser_mPair.second;

	cvtColor(mser_p, colorP, CV_GRAY2RGB);
	colorP2 = colorP.clone();
	colorP3 = colorP.clone();


	for (auto rect : bboxes_p)
	{
		cv::rectangle(colorP, rect, cv::Scalar(0, 0, 255), 1);
	}

	auto bboxes_p_pre = preDiscardBBoxes_p(bboxes_p, bboxes_m);




	for (auto rect : bboxes_p_pre)
	{
		cv::rectangle(colorP2, rect, cv::Scalar(0, 0, 255), 1);
	}


	visualize_p = colorP2;
	auto bboxes_p_real = realDiscardBBoxes_p(bboxes_p_pre, bboxes_m);
	visualize_p = colorP3;
	auto bboxes_p_post = postDiscardBBoxes_p(bboxes_p_real, bboxes_m); 

	std::vector<cv::Mat> candidates;

	for (auto rect : bboxes_p_real)
	{
		cv::rectangle(colorP3, relaxRect(rect), cv::Scalar(0, 255, 255), 1);
		cv::rectangle(img_bk, relaxRect(rect), cv::Scalar(0, 255, 255), 1);
	}

	for (auto rect : bboxes_p_post)
	{
		cv::rectangle(colorP3, rect, cv::Scalar(0, 255, 0), 1);
		cv::rectangle(img_bk, rect, cv::Scalar(0, 255, 0), 1);
		candidates.push_back(getROI(rect));
	}

	cvtColor(mser_m, colorM, CV_GRAY2RGB);
	for (cv::Rect rect : bboxes_m)
	{
		cv::rectangle(colorM, rect, cv::Scalar(0, 255, 255), 1);
	}


	if (DEBUG_LEVEL >= 1)
	{
		ImageViewer::viewImage(grey, "grey", 400);
		ImageViewer::viewImage(colorP3, "canidate mser_p", 400);
		ImageViewer::viewImage(mser_p, "response mser_p", 400);
		ImageViewer::viewImage(colorM, "response mser_m", 400);
	}

	ImageViewer::viewImage(img_bk, "candidates", 400);
	

	size_t i = 0;
	for (auto roi : candidates)
	{
		ImageViewer::viewImage(roi, "candidate " + std::to_string(i), 100);
		i++;
	}

	return candidates;
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

    cv::Ptr<cv::MSER> ptr = cv::MSER::create(5, 60, 14400, 0.25, 0.2, 100, 1.01, 0.003, 5); //all default values
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
	std::vector<std::pair<cv::Rect, int>> rectInnerElements = getNumInnerElements(boxes_p, boxes_v);

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

	cv::Mat reset_vis = visualize_p.clone();

	for (auto rect_p : boxes_p)
	{
		std::vector<cv::Rect> innerElements;
		std::vector<cv::Point2f> centerPoints;
		visualize_p = reset_vis.clone();
		cv::rectangle(visualize_p, rect_p, cv::Scalar(255, 0, 0), 1);
		for (auto rect_v : boxes_v)
		{
			if (intersectArea(rect_p, rect_v) == rect_v.area()) //rect_v completely in rect_p
			{
				innerElements.push_back(rect_v);
				centerPoints.push_back(cv::Point2f(rect_v.x + rect_v.width / 2, (rect_v.y + rect_v.height / 2)));
				cv::rectangle(visualize_p, rect_v, cv::Scalar(0, 255, 0), 1);
			}
		}

		if (centerPoints.size() == 0) continue; //really should not happen unless preprocess stage failed or was not called

		//STEP 1: fit line (robust)
		cv::Vec4f line;
		cv::fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01);
		float x0 = line[2];	float y0 = line[3]; float dx = line[0]; float dy = line[1];
		cv::line(visualize_p, cv::Point(x0 + dx * -1000, y0 + dy * -1000), cv::Point(x0 + dx * 1000, y0 + dy * 1000), cv::Scalar(255, 255, 0), 1);

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
			cv::rectangle(visualize_p, rect, cv::Scalar(0, 255, 255), 1);
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

	visualize_p = reset_vis.clone();
	return res;

}

std::tuple<double, double, double, double> MSER::meanStdDev(std::vector<cv::Rect> elems)
{
	auto power2 = [](double x) { return x * x; };
	uint sumWidth = 0;
	uint sumHeight = 0;

	for (auto rect : elems)
	{
		sumWidth += rect.width;
		sumHeight += rect.height;
	}

	double meanWidth = 1.0 * sumWidth / elems.size();
	double meanHeight = 1.0 * sumHeight / elems.size();

	double sqrSumWidth = 0; double sqrSumHeight = 0;
	for (auto rect : elems)
	{
		sqrSumWidth += power2(rect.width - meanWidth);
		sqrSumHeight += power2(rect.height - meanHeight);
	}

	double stdevWidth = std::sqrt(sqrSumWidth / elems.size());
	double stdevHeight = std::sqrt(sqrSumHeight / elems.size());

	return std::make_tuple(meanWidth, meanHeight, stdevWidth, stdevHeight);

}

//returns true if all almost same height and returns avg height, avg width
std::tuple<bool, float, float> MSER::sameSize(std::vector<cv::Rect> innerElements)
{
	auto tuple = MSER::meanStdDev(innerElements);
	double meanWidth = std::get<0>(tuple);
	double meanHeight = std::get<1>(tuple);
	double stdevWidth = std::get<2>(tuple);
	double stdevHeight = std::get<3>(tuple);

	int minHeight = -1; int maxHeight = -1;
	int minWidth = -1; int maxWidth = -1;
	double avgHeight = 0; double avgWidth = 0;

	double leftBoundary0 = meanHeight - 2 * stdevHeight;
	double rightBoundary0 = meanHeight + 2 * stdevHeight;
	double leftBoundary1 = meanWidth - 2 * stdevWidth;
	double rightBoundary1 = meanWidth + 2 * stdevWidth;

	size_t count = 0;
	for (auto elem : innerElements)
	{
		if (leftBoundary0 <= elem.height && elem.height <= rightBoundary0 && leftBoundary1 <= elem.width && elem.width <= rightBoundary1)
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


std::vector<std::pair<cv::Rect, int>> MSER::getNumInnerElements(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<std::pair<cv::Rect, int>> rectInnerElements;

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

		if (count > 2)
		{
			rectInnerElements.push_back(std::make_pair(rect_p, count));
		}
	}

	return rectInnerElements;

}


std::vector<cv::Rect> MSER::postDiscardBBoxes_p(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
{
	std::vector<std::pair<cv::Rect, int>> rectInnerElements = getNumInnerElements(boxes_p, boxes_v);

	for (auto elem : boxes_p)
	{
		cv::rectangle(visualize_p, relaxRect(elem), cv::Scalar(0, 255, 255), 1);
	}

	std::vector<std::pair<cv::Rect, int>> res;
	if (rectInnerElements.size() == 0) return std::vector<cv::Rect>();
	res.push_back(rectInnerElements[0]);

	cv::Mat reset_vis = visualize_p.clone();

	for (size_t i = 1; i < rectInnerElements.size(); i++)
	{
		auto elem1 = rectInnerElements[i];
		bool intersect = false;
		visualize_p = reset_vis.clone();
		cv::rectangle(visualize_p, relaxRect(elem1.first), cv::Scalar(0, 255, 0), 1);
		for (size_t j = 0; j < res.size(); j++)
		{
			auto elem2 = res[j];
			cv::rectangle(visualize_p, relaxRect(elem2.first), cv::Scalar(255, 255, 0), 1);

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
			else if (intersectArea(elem1.first, elem2.first) == elem1.first.area())
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
			else if ((intersectArea(elem1.first, elem2.first))) //overlapping but not real inner element
			{
				if (elem1.second > elem2.second)
					res[j] = elem1;
				else
					res[j] = elem2;
				intersect = true;
			}
		}

		if (!intersect)
			res.push_back(elem1);
	}

	std::vector<cv::Rect> res_flattened;
	for (auto elem : res)
	{
		//check if rect is too wide
		if (elem.first.width <= elem.first.height * MAX_ASPECT_RATIO &&
			elem.first.width > elem.first.height &&
			elem.first.width >= elem.first.height * MIN_ASPECT_RATIO)
		{
			//and relax borders by RELAX_PIXELS
			res_flattened.push_back(relaxRect(elem.first));
		}
			

	}

	return res_flattened;

}


cv::Rect MSER::relaxRect(cv::Rect rect)
{
	int w = RELAX_PIXELS;
	int x = (rect.x - w < 0) ? 0 : rect.x - w;
	int y = (rect.y - w < 0) ? 0 : rect.y - w;
	int width = (rect.x + rect.width + 2 * w > resizedImage.cols) ? resizedImage.cols - rect.x : rect.width + 2 * w;
	int height = (rect.y + rect.height + 2 * w > resizedImage.rows) ? resizedImage.rows - rect.y : rect.height + 2 * w;
	return cv::Rect(x, y, width, height);
}


cv::Mat MSER::getROI(cv::Rect rect)
{
	return originalImage(cv::Rect(rect.x / scaleFactor, rect.y / scaleFactor, rect.width / scaleFactor, rect.height / scaleFactor));
}

cv::Mat MSER::resizeImg(cv::Mat img)
{
	scaleFactor = 1;
	cv::Mat smallImg;;
	if (img.cols > 1100)
	{
		scaleFactor = 900.0 / img.cols;
		cv::resize(img, smallImg, cv::Size(), scaleFactor, scaleFactor, CV_INTER_AREA);
	}
	else
	{
		smallImg = img.clone();
	}
	return smallImg;
}

MSER::~MSER()
{
	//do not release originalImage because ROIs are References?
	grey.release();
	mser_p.release();
	mser_m.release();

	visualize_p.release();
	colorP.release();
	colorP2.release();
	colorP3.release();
	colorM.release();
	img_bk.release();
}
