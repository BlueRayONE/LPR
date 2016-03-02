#include "Wavelet.h"


template <typename T> std::vector<int> sort_indexi(const std::vector<T> &v)
{

	// initialize original index locations
	std::vector<int> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

	return idx;
}


Wavelet::Wavelet()
{
}

Wavelet::~Wavelet()
{
}

void Wavelet::run(cv::Mat img)
{
	//preprocess
	img = cv::Mat(img, cv::Range(0, (img.rows & ~3)), cv::Range(0, (img.cols & ~3)));

	if (img.rows > 800)
		SCALE_FACTOR = 2;

	cv::Mat grey;
	cv::cvtColor(img, grey, CV_BGR2GRAY);
	cv::equalizeHist(grey, grey);
	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(grey, "grey", 800);

	//generate Haar wavelet
	cv::Mat grey32F = cv::Mat(grey.size(), CV_32FC1);
	grey.convertTo(grey32F, CV_32FC1);
	grey32F = haarWavelet(grey32F, 1);
	cv::Mat haar;
	cv::convertScaleAbs(grey32F, haar);
	if(DEBUG_LEVEL == 2) ImageViewer::viewImage(haar, "Haar");


	//extract HL from Haar image
	cv::Mat haarHL = cv::Mat(haar.rows / 2, haar.cols / 2, CV_8U);
	for (int r = 0; r < haar.rows / 2; r++)
	{
		for (int c = haar.cols / 2; c < haar.cols; c++)
		{
			int current = c - haar.cols / 2;
			haarHL.at<uchar>(r, current) = haar.at<uchar>(r, c);
		}
	}
	

	if(DEBUG_LEVEL >= 1) ImageViewer::viewImage(haarHL, "HL");

	
	//binarize HL
	cv::Mat binarizedHL = binarize(haarHL);
	//if(FULL_DEBUG) ImageViewer::viewImage(binarizedHL, "HL bin");

	//morph HL
	cv::Mat morphedHL = haarHL;//this->morph(haarHL); //binarized
	if (DEBUG_LEVEL >= 1) ImageViewer::viewImage(morphedHL, "HL morphed");

	binarizedHL = binarize(morphedHL);

	cv::threshold(morphedHL, morphedHL, THRESH_TO_ZERO, 255, CV_THRESH_TOZERO);


	std::vector<int> binRS = this->calcRowSums(binarizedHL);
	std::vector<int> morphRS = this->calcRowSums(morphedHL);
	std::vector<int> haarRS = this->calcRowSums(haarHL);

	std::vector<float> gauss = this->gaussFilter(morphRS); //morphRS

	this->print(binRS);
	this->print(morphRS);
	this->print(haarRS);
	this->print(gauss);


	std::vector<std::pair<int,float>> peaks = this->findPeaks(gauss);
	std::pair<int, float> maxIDVal = this->findMaxPeak(peaks, gauss);
	float max = maxIDVal.second;
	std::vector<std::pair<int, int>> startRowsHeights =  this->findThresholdAreas(max, gauss);

	for (auto i = startRowsHeights.begin(); i != startRowsHeights.end(); ++i)
		qDebug() << (*i).first << " " << (*i).second;


/*only for debugging*/
/*************************************************************************************/
#pragma region draw_above_threshold
	cv::Mat colorHL2;
	cvtColor(haarHL, colorHL2, CV_GRAY2RGB);
	cv::Mat colorHL2_thres = colorHL2.clone();


	for (int r = 0; r < haarHL.rows; r++)
	{

		for (int c = 0; c < colorHL2.cols; c++)
		{
			if (gauss[r] > MAX_WEIGHT * max)
			{
				cv::Vec3b inten;
				inten.val[2] = 255;
				colorHL2_thres.at<cv::Vec3b>(r, c) = inten;
			}
		}
	}

    for(auto currRowHeightPair : startRowsHeights)
	{
        //auto current = startRowsHeights[i];
        for (int r = currRowHeightPair.first; r <= currRowHeightPair.first + currRowHeightPair.second; r++)
		{
			for (int c = 0; c < colorHL2.cols; c++)
			{
				cv::Vec3b inten;
				inten.val[1] = 255;
				colorHL2.at<cv::Vec3b>(r, c) = inten;
			}

		}
	}

	if (DEBUG_LEVEL >= 1)
		ImageViewer::viewImage(colorHL2, "draw thresholds adpative");
	if (DEBUG_LEVEL >= 2)
		ImageViewer::viewImage(colorHL2_thres, "draw thresholds");
#pragma endregion
/*************************************************************************************/
	
	
	std::vector<std::pair<float, cv::Rect>> candidatesRough = findRoughCandidate(morphedHL, startRowsHeights); //morphed
	candidatesRough = findNonIntCandidate(candidatesRough);
	if (candidatesRough.size() >= 10) candidatesRough.resize(10); //keep only first 10 candidates

/*only for debugging*/
/*************************************************************************************/
#pragma region draw_roughs
	cv::Mat colorHL;
	cv::Mat colorBinHL;
	cv::Mat colorMorphHL;
	cvtColor(haarHL, colorHL, CV_GRAY2RGB);
	cvtColor(binarizedHL, colorBinHL, CV_GRAY2RGB);
	cvtColor(morphedHL, colorMorphHL, CV_GRAY2RGB);
    for(auto pairWeightRect : candidatesRough)
	{
        cv::Rect currentRect = pairWeightRect.second;

		for (int r = currentRect.tl().y; r < currentRect.br().y; r++)
		{
			for (int c = currentRect.tl().x; c < currentRect.br().x; c++)
			{
				cv::Vec3b inten;
				inten.val[2] = 255;
				colorHL.at<cv::Vec3b>(r, c) = inten;

			}
		}
	}
	if (DEBUG_LEVEL >= 1) ImageViewer::viewImage(colorHL, "draw rough");
#pragma endregion
/*************************************************************************************/
	std::vector<std::pair<float, cv::Rect>> candidatesReal = findExactCandidate(grey, morphedHL, candidatesRough); //binarized

	std::sort(candidatesReal.begin(), candidatesReal.end(), [](const std::pair<float, cv::Rect> &left, const std::pair<float, cv::Rect> &right)
	{
		return left.first > right.first;
	});


/*only for debugging*/
/*************************************************************************************/
	std::vector<cv::Mat> matCandidates;
#pragma region draw_candidates
	//int count = (candidatesRough.size() > 10) ? 10 : candidatesRough.size();
    size_t count = candidatesRough.size();
    for (size_t i = 0; i <  candidatesRough.size(); i++)
	{
		
		if(i < count)
		{
			int r = 0;
			int g = 255;
            cv::Rect currentRect = candidatesRough[i].second;
			cv::Rect original = cv::Rect(currentRect.tl().x * 2 - 1, currentRect.tl().y * 2 - 1, currentRect.width * 2 + 2, currentRect.height * 2 + 2);

			cv::rectangle(colorHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(colorBinHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(colorMorphHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(img, original, cv::Scalar(255, 0, 0));

			
			if (i < candidatesReal.size())
			{

                cv::Rect currentRectEx = candidatesReal[i].second;
                qDebug() << candidatesReal[i].first;
				cv::Rect originalEx = cv::Rect(currentRectEx.tl().x * 2 - 1, currentRectEx.tl().y * 2 - 1, currentRectEx.width * 2 + 2, currentRectEx.height * 2 + 2);

				cv::rectangle(colorHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(colorBinHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(colorMorphHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(img, originalEx, cv::Scalar(0, g, r));
				matCandidates.push_back(img(cv::Rect(currentRectEx.tl().x * 2, currentRectEx.tl().y * 2, currentRectEx.width * 2, currentRectEx.height * 2)));
			}

		}
			
	}

#pragma endregion
/*************************************************************************************/


	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorHL, "colorHL");
	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorBinHL, "colorBinHL");
	if (DEBUG_LEVEL >= 0) ImageViewer::viewImage(img, "orig width candidates");
	
}


std::vector<float> Wavelet::gaussFilter(std::vector<int> arr)
{
	std::vector<float> res = std::vector<float>(arr.size(), 0.0);
	
	float sigma = 0.05f;
	int w = 2;


	auto h = [sigma](int j) { return exp(-((j*sigma)*(j*sigma))/2); };

	float k = 0;
	for (int j = -w; j <= w; j++)
	{
		k += h(j);
	}

	for (size_t i = w; i < arr.size() - w; i++)
	{
		float sum = 0;
		for (int j = -w; j <= w; j++)
		{
			sum += arr[i - j] * h(j);
		}
		sum /= k;
		res[i] = sum;
	}

	return res;

}


std::vector<int> Wavelet::calcRowSums(cv::Mat img)
{
	std::vector<int> sums = std::vector<int>(img.rows, 0);
 
	for (size_t r = 0; r < sums.size(); r++)
	{
		int currSum = 0;
		for (int c = 0; c < img.cols; c++)
		{
			currSum += img.at<uchar>(r, c);
		}

		sums[r] = currSum;

	}
	return sums;
}

std::vector<int> Wavelet::calcColSums(cv::Mat img)
{

	std::vector<int> sums = std::vector<int>(img.cols, 0);

	for (size_t c = 0; c < sums.size(); c++)
	{
		int currSum = 0;
		for (int r = 0; r < img.rows; r++)
		{
			currSum += img.at<uchar>(r, c);
		}

		sums[c] = currSum;
	}

	return sums;
}

cv::Mat Wavelet::haarWavelet(cv::Mat &src, int NIter)
{
	cv::Mat dst = cv::Mat(src.rows, src.cols, src.type());
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;
	for (int k = 0; k<NIter; k++)
	{
		for (int y = 0; y<(height >> (k + 1)); y++)
		{
			for (int x = 0; x<(width >> (k + 1)); x++)
			{
				c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x) = c;

				dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x + (width >> (k + 1))) = dh;

				dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x) = dv;

				dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			}
		}
		dst.copyTo(src);
	}

	return dst;
}

cv::Mat Wavelet::genGreyScale(cv::Mat img)
{
	int rows = img.rows;
	int cols = img.cols;
	cv::Mat greyScaleImage = cv::Mat::zeros(img.size(), CV_8U);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			cv::Vec3b bgr = img.at<cv::Vec3b>(r, c);
			greyScaleImage.at<uchar>(r, c) = (bgr[0]);
		}
	}

	return greyScaleImage;
}

cv::Mat Wavelet::binarize(cv::Mat img)
{
	cv::Mat res;
	cv::threshold(img, res, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	return res;
}

cv::Mat Wavelet::morph(cv::Mat img)
{
	cv::Mat res, temp;
	int element_shape = 0;
	cv::Mat element = cv::getStructuringElement(element_shape, cv::Size(3, 1));
	cv::morphologyEx(img, res, 6, element);
	return res;
}

void Wavelet::filterNeighbours(cv::Mat img)
{
	for (int r = 1; r < img.rows - 1; r++)
	{
		for (int c = 1; c < img.cols - 1; c++)
		{
			int count = 0;

			if (img.at<uchar>(r, c) != 0)
			{
				if (img.at<uchar>(r - 1, c - 1) == 0)
					count++;
				if (img.at<uchar>(r - 1, c) == 0)
					count++;
				if (img.at<uchar>(r - 1, c + 1) == 0)
					count++;
				if (img.at<uchar>(r, c - 1) == 0)
					count++;
				if (img.at<uchar>(r, c + 1) == 0)
					count++;
				if (img.at<uchar>(r + 1, c - 1) == 0)
					count++;
				if (img.at<uchar>(r + 1, c) == 0)
					count++;
				if (img.at<uchar>(r + 1, c + 1) == 0)
					count++;
			}

			if (count == 8)
				img.at<uchar>(r, c) = 0;

		}
	}
}

std::vector<std::pair<int, float>> Wavelet::findPeaks(std::vector<float> arr)
{
	std::vector<std::pair<int, float>> peaks;
	std::vector<float> firstDeriv(arr.size(), 0.0);

	for (size_t i = 1; i < arr.size(); i++)
	{
		firstDeriv[i] = arr[i] - arr[i - 1];
	}

	
	bool sign = false;

	for (size_t i = 1; i < arr.size(); i++)
	{
		if (firstDeriv[i] < 0)
		{
			if (sign == false) //was negative is postive --> Max
			{
				peaks.push_back(std::make_pair(i , arr[i]));
				sign = true;
			}

		}
		else if (firstDeriv[i] > 0)
		{
			sign = false;
		}
	}

	return peaks;

}

std::pair<int, float> Wavelet::findMaxPeak(std::vector<std::pair<int, float>> peaks, std::vector<float> arr)
{
	float max = 0;
	if (peaks.size() > 0)
	{
		max = peaks[0].second;
	}
	int maxid = 0;


    for (auto currentPairIdMax : peaks)
	{
        if (currentPairIdMax.second > max)
		{
            max = currentPairIdMax.second;
            maxid = currentPairIdMax.first;
		}
	}

	return std::make_pair(maxid, max);
}

float Wavelet::rectRank(cv::Mat img, cv::Rect rect)
{
	cv::Mat debug = img.clone()(rect);


	cv::Point tl = rect.tl();
	float rectWeight = 0.0;
	

	for (int j = tl.x; j < tl.x + rect.width; j++)
	{
		for (int i = tl.y; i < tl.y + rect.height; i++)
		{
			int val = img.at<uchar>(i, j);
			if (val != 0)
			{
				rectWeight += val;
			}
			debug.at<uchar>(i-tl.y, j-tl.x) = 255;
			
		}


	}
	rectWeight /= rect.area();

	return rectWeight;

}

std::vector<std::pair<int, int>> Wavelet::findThresholdAreas(double max, std::vector<float> rowSums, bool splitAreas)
{
	std::vector<std::pair<int, int>> startRowsHeights;

	const int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * rowSums.size());
	float currentWeight = MAX_WEIGHT;

	int currentHeight = 0;
	int currentRectStart = 0;
	bool wasInThresh = false;
	bool isInThresh = false;


	for (size_t r = 0; r < rowSums.size(); r++)
	{
		if (rowSums[r] >= currentWeight * max)
			isInThresh = true;
		

		if (isInThresh)
		{
			if (!wasInThresh)
				currentRectStart = r;
			currentHeight++;

			if (r == rowSums.size() - 1)
				isInThresh = false;
		}
			
		else if (wasInThresh && !isInThresh)
		{
			if (currentHeight > maxRectHeight && splitAreas)
			{
				r = currentRectStart - 1; //revert rectangle and refine threshold
				currentWeight += WEIGHT_STEP_SIZE;
				isInThresh = false;
				wasInThresh = false;
				currentHeight = 0;
				continue;
			}

			if (currentHeight > MIN_RECT_HEIGHT_RATIO * rowSums.size() || !splitAreas)
			{
				startRowsHeights.push_back(std::make_pair(r - currentHeight, currentHeight));
			}
			currentWeight = MAX_WEIGHT; //reset currentWeight for next retangles;
			currentHeight = 0;
		}

		wasInThresh = isInThresh;
		isInThresh = false;
	}

	return startRowsHeights;
}

std::vector<std::pair<float, cv::Rect>> Wavelet::findRoughCandidate(cv::Mat img, std::vector<std::pair<int, int>> startRowsHeights)
{
	//breite: 520 hoehe: 110
	cv::Mat debug, debug2;
	debug2 = img.clone();
	std::vector<std::pair<float, cv::Rect>> candidates;
	int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * img.rows);

    for (auto currPairRowHeight : startRowsHeights) //number of candidates
	{
		debug = img.clone();
        cv::Rect debugRect = cv::Rect(0, currPairRowHeight.first, img.rows, currPairRowHeight.second);
		debug = debug(debugRect);
        int startRow = currPairRowHeight.first;
        int height = currPairRowHeight.second;
		int width = height * HEIGHT_TO_WIDTH_RATIO;


		for (int c = 0; c < img.cols - width; c++)
		{
			cv::Rect current = cv::Rect(c, startRow, width, height);
			cv::Point center = cv::Point(c + width / 2, startRow + height / 2);
			cv::rectangle(debug2, current, cv::Scalar(255, 0, 0));
			if (img.at<uchar>(center) == 0 && img.at<uchar>(cv::Point(center.x, center.y + 1)) == 0 && img.at<uchar>(cv::Point(center.x, center.y - 1)) == 0)
				continue;

			
			float rank = this->rectRank(img, current);
			if (evalRect(current, img))
				candidates.push_back(std::make_pair(rank, current));
		}
	}

	maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * img.rows);


	return candidates;

}

//finds (max) n distinct non intersecting rectangles
std::vector<std::pair<float, cv::Rect>> Wavelet::findNonIntCandidate(std::vector<std::pair<float, cv::Rect>> candidates)
{
    //push first, goto next, if intersects one in list and has higher weight replace!!
    //return first n
    std::vector<std::pair<float, cv::Rect>> res;

    if (candidates.size() == 0)
        return candidates;

    res.push_back(candidates[0]);

    for (size_t i = 1; i < candidates.size(); i++)
    {
        bool intersect = false;

        for (size_t k = 0; k < res.size(); k++)
        {
            //check if existing candidates intersect with new one and replace if new one has better weight
            if (this->rectIntersect(candidates[i].second, res[k].second))
            {
                intersect = true;
                if(candidates[i].first > res[k].first)
                    res[k] = candidates[i]; //replace existing with new
            }
        }

        if (!intersect)
            res.push_back(candidates[i]);

    }

    //sort by highest weight (heightest weight first)
    std::sort(res.begin(), res.end(), [](const std::pair<float, cv::Rect> &left, const std::pair<float, cv::Rect> &right)
    {
        return left.first > right.first;
    });

    return res;

}

std::vector<std::pair<float, cv::Rect>> Wavelet::findExactCandidate(cv::Mat grey, cv::Mat rankImg,  std::vector<std::pair<float, cv::Rect>> candidates)
{
	std::vector<std::pair<float, cv::Rect>> res;
	
    for (auto pairWeightRect : candidates)
	{
        cv::Rect currentRect = pairWeightRect.second;
		cv::Rect originalRect = cv::Rect(currentRect.tl().x * 2, currentRect.tl().y * 2, currentRect.width * 2, currentRect.height * 2);

		cv::Mat candidate = grey(originalRect).clone();

		cv::Mat grey32F = cv::Mat(candidate.size(), CV_32FC1);
		candidate.convertTo(grey32F, CV_32FC1);

		grey32F = haarWavelet(grey32F, 1);
		cv::Mat candidateHaar;
		cv::convertScaleAbs(grey32F, candidateHaar);
		
		if (DEBUG_LEVEL >= 1) ImageViewer::viewImage(candidateHaar, "Haar2");

		cv::Mat candidateHL = cv::Mat(candidateHaar.rows / 2, candidateHaar.cols / 2, CV_8U);
        for (int r = 0; r < candidateHaar.rows / 2; r++)
		{
			for (int c = candidateHaar.cols / 2; c < candidateHaar.cols; c++)
			{
				int current = c - candidateHaar.cols / 2;
				candidateHL.at<uchar>(r, current) = candidateHaar.at<uchar>(r, c);
			}
		}

		candidate.release(); grey32F.release(); candidateHaar.release();


		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(candidateHL.rows * GAP_TO_HEIGHT_RATIO, candidateHL.rows));
		cv::morphologyEx(candidateHL, candidateHL, 3, element); //3 == closing

		element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1)); //dilate 9x1 to compensate cut offs
		cv::dilate(candidateHL, candidateHL, element);


		std::vector<int> colsSums = this->calcColSums(candidateHL);
		std::vector<float> gaussColsSums = this->gaussFilter(colsSums);
		this->print<float>(gaussColsSums);

		std::vector<std::pair<int, float>> peaks = this->findPeaks(gaussColsSums);
		std::pair<int, float> maxIDVal = this->findMaxPeak(peaks, gaussColsSums);
		float max = maxIDVal.second;


		int pos_left = 0;
		int pos_right = candidateHL.cols - 1;

		for (int i = 0; i < candidateHL.cols; i++)
		{

			if (gaussColsSums[i] > 0.5 * max)
			{
				pos_left = (i == 2) ? 0 : i;
				break;
			}
		}

		for (int i = candidateHL.cols - 1; i >= 0; i--)
		{

			if (gaussColsSums[i] > 0.5 * max)
			{
				pos_right = (i == candidateHL.cols - 3) ? candidateHL.cols - 1 : i;
				break;
			}
		}
		cv::Rect rect = cv::Rect(pos_left, 0, pos_right - pos_left + 1, candidateHL.rows);

		rect += currentRect.tl(); //translate back
		float weight = this->rectRank(rankImg, rect); 

		if (!evalRect(rect, rankImg)) continue;
		res.push_back(std::make_pair(weight, rect));

		
	}

	return res;
}

//returns true if rect satisfies candidate properties
bool Wavelet::evalRect(cv::Rect rect, cv::Mat evalImg)
{
	if (rect.width <= rect.height * 1.5) //doesnt work for usa license plates
		return false;
	if (rect.width < rect.height)
		return false;
	if (rect.height * GAP_TO_HEIGHT_RATIO >= rect.width)
		return false;
	if (rect.height * GAP_TO_HEIGHT_RATIO < 1)
		return false;
	

	return true;
}

bool  Wavelet::rectIntersect(cv::Rect r1, cv::Rect r2)
{
	return  (r1 & r2).area();

}

template<typename T> void Wavelet::print(std::vector<T> arr)
{
	QString res = "[";
	for (auto const& elem : arr)
	{
		res += QString::number(elem) + " ";
	}

	res += "]";
	qDebug() << res;
}
