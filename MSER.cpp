#include "MSER.h"


MSER::MSER()
{
}


cv::Mat global;

void MSER::run(cv::Mat img)
{
	cv::Mat grey, grey_inv, mser_p, mser_m;

	cv::cvtColor(img, grey, CV_BGR2GRAY);
	cv::Ptr<cv::MSER> ptr = cv::MSER::create(5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5);
	ptr->setPass2Only(true);


	//cv::Mat ellipses = originalImage.clone();

	cv::bitwise_not(grey, grey_inv);

	mser_p = cv::Mat(grey.rows, grey.cols, CV_8U, cv::Scalar(0));
	mser_m = cv::Mat(grey.rows, grey.cols, CV_8U, cv::Scalar(0));



	std::vector< std::vector< cv::Point > > msers_p;
	std::vector< cv::Rect > bboxes_p;
	ptr->detectRegions(grey, msers_p, bboxes_p);

	std::vector< std::vector< cv::Point > > msers_m;
	std::vector< cv::Rect  > bboxes_m;
	ptr->detectRegions(grey_inv, msers_m, bboxes_m);


	for (std::vector<cv::Point> v : msers_p)
	{
		for (cv::Point pt : v)
		{
			mser_p.at<uchar>(pt) = 255;
		}
	}

	for (std::vector<cv::Point> v : msers_m)
	{
		for (cv::Point pt : v)
		{
			mser_m.at<uchar>(pt) = 255;
		}
	}


	cv::Mat mser_p_bk = mser_p.clone();
	//keep non intersecting biggest rectangles, weigh more width than height more

	for (cv::Rect rect : bboxes_p)
	{
		cv::rectangle(mser_p_bk, rect, cv::Scalar(255), 1);
	}


	cv::Mat colorP;
	cvtColor(mser_p, colorP, CV_GRAY2RGB);

	auto bboxes_p_pre = preDiscardBBoxes_p(bboxes_p, bboxes_m);

	for (auto rect : bboxes_p_pre)
	{
		cv::rectangle(colorP, rect, cv::Scalar(0, 0, 255), 1);
	}

	global = colorP;

	auto bboxes_p_real = realDiscardBBoxes_p(bboxes_p_pre, bboxes_m);




	for (auto rect : bboxes_p_real)
	{
		cv::rectangle(colorP, rect, cv::Scalar(0, 255, 0), 1);
	}

	for (cv::Rect rect : bboxes_m)
	{
		cv::rectangle(mser_m, rect, cv::Scalar(255), 1);
	}



	ImageViewer::viewImage(grey, "grey", 400);
	ImageViewer::viewImage(grey_inv, "grey_inv", 400);

	ImageViewer::viewImage(colorP, "response mser_p", 400);
	ImageViewer::viewImage(mser_p_bk, "response mser_p_bk", 400);
	ImageViewer::viewImage(mser_m, "response mser_m", 400);
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

		cv::Vec4f line;
		cv::fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01);

		float x0 = line[2];
		float y0 = line[3];
		float x1 = line[0];
		float y1 = line[1];

		cv::line(global, cv::Point(x0 + x1 * -1000, y0 + y1 * -1000), cv::Point(x0 + x1 * 1000, y0 + y1 * 1000), cv::Scalar(255, 255, 0), 1);

		auto tuple = sameSize(innerElements); //tuple(bool sameSize, avgHeight, avgWidth)

		//STEP 1: check if MSER- bboxes equal in size
		if (!std::get<0>(tuple)) //not same Height --> discard rect_p
			continue;

		//STEP 2: check if MSER+ bbox almost same size as average height
		if (rect_p.height - MAX_BBOX_HEIGHT_SCALE * std::get<1>(tuple) > 0)
			continue;
		
		//maybe do
		/*if (rect_p.width - innerElements.size() * std::get<2>(tuple) > MAX_BBOX_WIDTH_VARIANCE)
			continue;*/


		//STEP 3: check if center points of MSER- Elements lie on same line


		//all steps passed --> push back
		res.push_back(rect_p);


	}

	return res;

}


//returns true if all almost same height and returns avg height, avg width
std::tuple<bool, float, float> MSER::sameSize(std::vector<cv::Rect> innerElements)
{
	int minHeight = -1; int maxHeight = -1;
	int minWidth = -1; int maxWidth = -1;
	float avgHeight = 0; float avgWidth = 0;

	if (innerElements.size() > 0)
	{
		maxHeight = innerElements[0].height;
		maxWidth = innerElements[0].width;
		minHeight = innerElements[0].height;
		minWidth = innerElements[0].width;
	}
		

	for (auto elem : innerElements)
	{
		if (elem.height > maxHeight)
			maxHeight = elem.height;
		if (elem.height < minHeight)
			minHeight = elem.height;

		if (elem.width > maxWidth)
			maxWidth = elem.width;
		if (elem.width < minWidth)
			minWidth = elem.width;

		avgHeight += elem.height; avgWidth += elem.width;
	}

	avgHeight /= innerElements.size(); avgWidth /= innerElements.size();

	bool equal = false;
	if (maxHeight <= MAX_HEIGHT_SCALE * minHeight && maxWidth <= MAX_WIDTH_SCALE * minWidth)
		equal = true;

	return std::make_tuple(equal, avgHeight, avgWidth);
	
}


std::vector<cv::Rect> MSER::preDiscardBBoxes_p(std::vector<cv::Rect> boxes_p, std::vector<cv::Rect> boxes_v)
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


	std::vector<std::pair<cv::Rect, int>> res;
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


MSER::~MSER()
{
}
