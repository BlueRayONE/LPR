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
	img = cv::Mat(img, cv::Range(0, (img.rows & ~1)), cv::Range(0, (img.cols & ~1)));

	cv::Mat grey;
	cv::cvtColor(img, grey, CV_BGR2GRAY);
	cv::equalizeHist(grey, grey);

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


	int* binRS = this->calcRowSums(binarizedHL);
	int* morphRS = this->calcRowSums(morphedHL);
	int* haarRS = this->calcRowSums(haarHL);

	float* gauss = this->gaussFilter(morphRS, binarizedHL.rows); //morphRS

	this->print(binRS, binarizedHL.rows);
	this->print(morphRS, binarizedHL.rows);
	this->print(haarRS, binarizedHL.rows);
	this->print(gauss, binarizedHL.rows);


	std::vector<std::pair<int,float>> peaks = this->findPeaks(gauss, binarizedHL.rows);
	std::pair<int, float> maxIDVal = this->findMaxPeak(peaks, gauss);
	int maxID = maxIDVal.first;
	float max = maxIDVal.second;

	double avg = std::accumulate(gauss, gauss + binarizedHL.rows, 0.0) / (double) (binarizedHL.rows);
	avg = max;

	std::vector<std::pair<int, int>> startRowsHeights =  this->findThresholdAreas(haarHL.rows, avg, gauss);

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
			if (gauss[r] > AVG_WEIGHT*avg)
			{
				cv::Vec3b inten;
				inten.val[2] = 255;
				colorHL2_thres.at<cv::Vec3b>(r, c) = inten;
			}
		}
	}

	for (int i = 0; i < startRowsHeights.size(); i++)
	{
		auto current = startRowsHeights[i];
		for (int r = current.first; r <= current.first + current.second; r++)
		{
			for (int c = 0; c < colorHL2.cols; c++)
			{
				cv::Vec3b inten;
				inten.val[1] = 255;
				colorHL2.at<cv::Vec3b>(r, c) = inten;
			}

		}
	}

	if (DEBUG_LEVEL >= 2)
		ImageViewer::viewImage(colorHL2, "draw thresholds adpative");
	if (DEBUG_LEVEL >= 3)
		ImageViewer::viewImage(colorHL2_thres, "draw thresholds");
#pragma endregion
/*************************************************************************************/

	//check a fre neighbours maybe and use binarized
	cv::Mat bla;
	cv::medianBlur(binarizedHL, bla, 3);
	
	
	std::vector<std::pair<float, cv::Rect>> candidatesRough = findRoughCandidate(morphedHL, startRowsHeights); //morphed
	candidatesRough = findNonIntCandidate(candidatesRough);
	if (candidatesRough.size() >= 10) candidatesRough.resize(10);

/*only for debugging*/
/*************************************************************************************/
#pragma region draw_roughs
	cv::Mat colorHL;
	cv::Mat colorBinHL;
	cv::Mat colorMorphHL;
	cvtColor(haarHL, colorHL, CV_GRAY2RGB);
	cvtColor(binarizedHL, colorBinHL, CV_GRAY2RGB);
	cvtColor(morphedHL, colorMorphHL, CV_GRAY2RGB);
	for (int i = 0; i < candidatesRough.size(); i++)
	{
		cv::Rect currentRect = candidatesRough[i].second;

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
#pragma region draw_peaks
	/*std::vector<int> v1;
	std::vector<int> v2;

	for (int i = 0; i < binarizedHL.rows; i++)
	{
		//v1.push_back(vals2[i]);
		v1.push_back((int)gauss[i]); //v1 smoothed morph
		v2.push_back(binRS[i]);		//binarized
	}
	
	for (int i = 0; i < peakIDs.size(); i++)
	{
		int row1 = sort_indexi(v1)[i];
		row1 = peakIDs[i];
		int row2 = sort_indexi(v2)[i];

		for (int c = 0; c < haarHL.cols; c++)
		{
			cv::Vec3b inten;
			inten.val[0] = 255;
			colorHL.at<cv::Vec3b>(row1, c) = inten; //morphed
			inten.val[1] = 255;
			colorHL.at<cv::Vec3b>(row2, c) = inten; //non morphed
		}

		for (int c = 0; c < img.cols; c++)
		{
			cv::Vec3b inten;
			inten.val[2] = 255;
			inten.val[1] = mvAvg[row1] * 255 / max;
			img.at<cv::Vec3b>(2*row1, c) = inten;
			inten.val[1] = 255;
			img.at<cv::Vec3b>(2 * row2, c) = inten;
		}
	}*/
#pragma endregion
/*************************************************************************************/


/*only for debugging*/
/*************************************************************************************/
#pragma region draw_candidates
	cv::Mat test;
	//int count = (candidatesRough.size() > 10) ? 10 : candidatesRough.size();
	int count = candidatesRough.size();
	for (int i = 0; i < candidatesRough.size(); i++)
	{
		
		if(i < count)
		{

			//int r = 0 + (i + 1)*1.0 / count * 255;
			//int g = 255 - (i + 1)*1.0 / count * 255;
			int r = 0;
			int g = 255;

			/*cv::rectangle(colorHL, candidatesRough[i].second, cv::Scalar(0, g, r));
			cv::rectangle(colorBinHL, candidatesRough[i].second, cv::Scalar(0, g, r));
			cv::rectangle(colorMorphHL, candidatesRough[i].second, cv::Scalar(0, g, r));
			cv::rectangle(img, original, cv::Scalar(0, g, r));*/
			cv::Rect currentRect = candidatesRough[i].second;
			cv::Rect original = cv::Rect(currentRect.tl().x * 2, currentRect.tl().y * 2, currentRect.width * 2, currentRect.height * 2);

			cv::rectangle(colorHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(colorBinHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(colorMorphHL, currentRect, cv::Scalar(255, 0, 0));
			cv::rectangle(img, original, cv::Scalar(255, 0, 0));

			
			if (i < candidatesReal.size())
			{

				cv::Rect currentRectEx = candidatesReal[i].second;
				cv::Rect originalEx = cv::Rect(currentRectEx.tl().x * 2, currentRectEx.tl().y * 2, currentRectEx.width * 2, currentRectEx.height * 2);

				cv::rectangle(colorHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(colorBinHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(colorMorphHL, currentRectEx, cv::Scalar(0, g, r));
				cv::rectangle(img, originalEx, cv::Scalar(0, g, r));

				/*ONLY TEMPORARILY*/
				if (i == 0) //extract first candidate
				{
					test = grey(originalEx);
				}
			}

		}
			
	}

#pragma endregion
/*************************************************************************************/


	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorHL, "colorHL");
	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorBinHL, "colorBinHL");
	//if (DEBUG_LEVEL >= 1) ImageViewer::viewImage(colorMorphHL, "colorMorphHL");
	if (DEBUG_LEVEL >= 0) ImageViewer::viewImage(img, "orig width candidates");
	if (DEBUG_LEVEL >= 0) ImageViewer::viewImage(test, "candidate");



	
}


float* Wavelet::gaussFilter(int* arr, int n)
{
	float* res = new float[n];
	
	res[0] = 0; res[1] = 0; res[2] = 0; res[3] = 0;
	res[n-1] = 0; res[n-2] = 0; res[n-3] = 0; res[n-4] = 0;
	float sigma = 0.05f;
	int w = 2;

	auto h = [sigma](int j) { return exp(-((j*sigma)*(j*sigma))/2); };

	float k = 0;
	for (int j = -w; j <= w; j++)
	{
		k += h(j);
	}

	for (int i = w; i < n - w; i++)
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

float* Wavelet::movingAvg(float* arr, int n)
{
	float* sums = new float[n + 1];
	float* res = new float[n];
	float sum = 0;
	sums[0] = 0;
	
	for (int i = 1; i < n + 1; i++)
	{
		sum += arr[i - 1];
		sums[i] = sum;
	}

	res[0] = 0; res[1] = 0; res[n-1] = 0; res[n-2] = 0;
	for (int i = 3; i < n - 1; i++)
	{
		res[i - 1] = (sums[i + 2] - sums[i - 3]) / 5;
	}

	delete sums;
	return res;
}

int* Wavelet::calcRowSums(cv::Mat img)
{
	int* sums = new int[img.rows];
	for (int r = 0; r < img.rows; r++)
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

int* Wavelet::calcColSums(cv::Mat img)
{
	int* sums = new int[img.cols];
	for (int c = 0; c < img.cols; c++)
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
			//greyScaleImage.at<uchar>(r, c) = (3 * bgr[2] + 6 * bgr[1] +  bgr[0])/10; //using luminosity method
			greyScaleImage.at<uchar>(r, c) = (bgr[0]);
		}
	}

	return greyScaleImage;
}

cv::Mat Wavelet::binarize(cv::Mat img)
{
	cv::Mat res;

	//otsu
	cv::threshold(img, res, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	return res;

}

cv::Mat Wavelet::morph(cv::Mat img)
{
	cv::Mat res, temp;

	/*int element_shape = 0;
	cv::Mat element = cv::getStructuringElement(element_shape, cv::Size(3, 1));


	cv::morphologyEx(img, res, 6, element);*/

	int element_shape = 0;
	cv::Mat element = cv::getStructuringElement(element_shape, cv::Size(3, 1));


	cv::morphologyEx(img, res, 6, element);



	return res;
}



std::vector<std::pair<int, float>> Wavelet::findPeaks(float* arr, int n)
{
	std::vector<std::pair<int, float>> peaks;

	float* firstDeriv = new float[n];
	firstDeriv[0] = 0;
	for (int i = 1; i < n; i++)
	{
		firstDeriv[i] = arr[i] - arr[i - 1];
	}

	
	bool sign = false;

	for (int i = 1; i < n; i++)
	{
		if (firstDeriv[i] < 0)
		{
			if (sign == false) //was negative is postive --> Max
			{
				peaks.push_back(std::make_pair(i , arr[i]));
				//qDebug() << i;
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

std::pair<int, float> Wavelet::findMaxPeak(std::vector<std::pair<int, float>> peaks, float* arr)
{
	float max = 0;
	if (peaks.size() > 0)
	{
		//max = arr[peakIDs[0]];
		max = peaks[0].second;
	}
	int maxid = 0;


	for (int i = 1; i < peaks.size(); i++)
	{
		if (peaks[i].second > max)
		{
			max = peaks[i].second;
			maxid = peaks[i].first;
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
		;
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

std::vector<std::pair<int, int>>  Wavelet::findThresholdAreas(int n, double avg, float* rowSums, bool splitAreas)
{
	std::vector<std::pair<int, int>> startRowsHeights;

	const int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * n);
	float currentWeight = AVG_WEIGHT;

	int currentHeight = 0;
	int currentRectStart = 0;
	bool wasInThresh = false;
	bool isInThresh = false;


	for (int r = 0; r < n; r++)
	{
		if (rowSums[r] >= currentWeight*avg)
			isInThresh = true;
		

		if (isInThresh)
		{
			if (!wasInThresh)
				currentRectStart = r;
			currentHeight++;

			if (r == n - 1)
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

			if (currentHeight > MIN_RECT_HEIGHT_RATIO * n || !splitAreas)
			{
				startRowsHeights.push_back(std::make_pair(r - currentHeight, currentHeight));
			}
			currentWeight = AVG_WEIGHT; //reset currentWeight for next retangles;
			currentHeight = 0;
		}

		wasInThresh = isInThresh;
		isInThresh = false;
	}

	return startRowsHeights;
}


/*std::vector<std::pair<int, int>>  Wavelet::findThresholdAreas(int n, double avg, float* rowSums)
{
	int currentHeight = 0;
	bool wasInThresh = false;
	bool isInThresh = false;
	int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * n);

	std::vector<std::pair<int, int>> startRowsHeights;

	for (int r = 0; r < n; r++)
	{
		if (rowSums[r] >= AVG_WEIGHT*avg)
			isInThresh = true;

		if (isInThresh)
		{
			currentHeight++;
			if (r == n - 1 && wasInThresh)
			{
				if (currentHeight > MIN_RECT_HEIGHT_RATIO * n)
				{
					startRowsHeights.push_back(std::make_pair(r - currentHeight, currentHeight));

				}

			}
		}

		else if (wasInThresh && !isInThresh)
		{
			if (currentHeight > MIN_RECT_HEIGHT_RATIO * n)
			{
				startRowsHeights.push_back(std::make_pair(r - currentHeight, currentHeight));

			}
			currentHeight = 0;
		}

		wasInThresh = isInThresh;
		isInThresh = false;
	}

	return startRowsHeights;
}*/

std::vector<std::pair<float, cv::Rect>>  Wavelet::findRoughCandidate(cv::Mat img, std::vector<std::pair<int, int>> startRowsHeights)
{
	//breite: 520 hoehe: 110
	cv::Mat debug, debug2;
	debug2 = img.clone();
	std::vector<std::pair<float, cv::Rect>> candidates;
	int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * img.rows);

	for (int i = 0; i < startRowsHeights.size(); i++) //number of candidates
	{
		debug = img.clone();
		cv::Rect debugRect = cv::Rect(0, startRowsHeights[i].first, img.rows, startRowsHeights[i].second);
		debug = debug(debugRect);
		int startRow = startRowsHeights[i].first;
		int height = startRowsHeights[i].second;
		int width = height * 5;


		for (int c = 0; c < img.cols - width; c++)
		{
			cv::Rect current = cv::Rect(c, startRow, width, height);
			cv::Point center = cv::Point(c + width / 2, startRow + height / 2);
			cv::rectangle(debug2, current, cv::Scalar(255, 0, 0));
			if (img.at<uchar>(center) == 0 && img.at<uchar>(cv::Point(center.x, center.y + 1)) == 0 && img.at<uchar>(cv::Point(center.x, center.y - 1)) == 0)
				continue;

			
			float rank = this->rectRank(img, current);
			if (evalRect(current, rank, img))
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

	for (int i = 1; i < candidates.size(); i++)
	{
		cv::Rect currentRect = candidates[i].second;
		bool intersect = false;

		for (int k = 0; k < res.size(); k++)
		{
			cv::Rect compareRect = res[k].second;
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

	/*int count = 0;
	for (int i = 0; i < candidates.size(); i++)
	{
		std::vector<std::pair<float, cv::Rect>> current_intersecting;
		cv::Rect currentRect = candidates[i].second;
		bool intersect = false;
		for (int k = 0; k < i; k++)
		{
			if (this->rectIntersect(candidates[k].second, currentRect))
			{
				intersect = true;
				//current_intersecting
				break;
			}
				
		}

		if (count < n && !intersect)
		{
			res.push_back(candidates[i]);
			count++;

		}

	}*/

	return res;

}

std::vector<std::pair<float, cv::Rect>> Wavelet::findExactCandidate(cv::Mat grey, cv::Mat rankImg,  std::vector<std::pair<float, cv::Rect>> candidates)
{
	std::vector<std::pair<float, cv::Rect>> res;
	/*for (int i = 0; i < candidates.size(); i++)
	{
		cv::Rect currentRect = candidates[i].second;
		float currentWeight = candidates[i].first;

		//reduce width --> first right then left
		cv::Rect newRect;
		for (int w = currentRect.width - 1; w >= 1; w--)
		{
			cv::Rect newRect = cv::Rect(currentRect.x, currentRect.y, w, currentRect.height);
			float newWeight = this->rectRank(img, newRect);

			if (newWeight >= currentWeight)
			{
				currentRect = newRect;
				currentWeight = newWeight;
			}
			else
				break;
		}

		for (int x = currentRect.tl().x; x < currentRect.br().x; x++)
		{
			cv::Rect newRect = cv::Rect(cv::Point(x, currentRect.tl().y), currentRect.br());
			float newWeight = this->rectRank(img, newRect);

			if (newWeight >= currentWeight)
			{
				currentRect = newRect;
				currentWeight = newWeight;
			}
			else
				break;

		}

		res.push_back(std::make_pair(currentWeight, currentRect));


	}*/

	for (int i = 0; i < candidates.size(); i++)
	{
		cv::Rect currentRect = candidates[i].second;
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


		int* colsSums = this->calcColSums(candidateHL);
		float* gaussColsSums = this->gaussFilter(colsSums, candidateHL.cols);
		this->print<float>(gaussColsSums, candidateHL.cols);

		std::vector<std::pair<int, float>> peaks = this->findPeaks(gaussColsSums, candidateHL.cols);
		std::pair<int, float> maxIDVal = this->findMaxPeak(peaks, gaussColsSums);
		int maxID = maxIDVal.first;
		float max = maxIDVal.second;


		double avg = std::accumulate(gaussColsSums, gaussColsSums + candidateHL.cols, 0.0) / (double) (candidateHL.cols);
		avg = 1.0*avg;
		avg = 0.5*max;

		int pos_left = 0;
		int pos_right = candidateHL.cols - 1;

		for (int i = 0; i < candidateHL.cols; i++)
		{

			if (gaussColsSums[i] > avg)
			{
				pos_left = i;
				break;
			}
		}

		for (int i = candidateHL.cols - 1; i >= 0; i--)
		{

			if (gaussColsSums[i] > avg)
			{
				pos_right = i;
				break;
			}
		}

		//std::vector<std::pair<int, int>> startColsWidths = this->findThresholdAreas(candidateHL.cols, avg, gaussColsSums, false);

		/*auto maxIt = std::max_element(startColsWidths.begin(), startColsWidths.end(),
			[](const std::pair<int, int>& p1, const std::pair<int, int>& p2)
			{
				return p1.second < p2.second;
			}
		);

		//(*maxit).first = pos   (*maxit).second = width

		for (auto i = startColsWidths.begin(); i != startColsWidths.end(); ++i)
			qDebug() << (*i).first << " " << (*i).second;

		cv::Rect rect = cv::Rect((*maxIt).first, 0, (*maxIt).second, candidateHL.rows);*/
		cv::Rect rect = cv::Rect(pos_left, 0, pos_right - pos_left + 1, candidateHL.rows);

		rect += currentRect.tl(); //translate back
		float weight = this->rectRank(rankImg, rect); 

		if (!evalRect(rect, weight, rankImg)) continue;
		res.push_back(std::make_pair(weight, rect));

		
	}

	return res;
}

//returns true if rect satisfies candidate properties
bool Wavelet::evalRect(cv::Rect rect, float rank, cv::Mat evalImg)
{
	/*if (rect.width <= rect.height * 2) //doesnt work for usa license plates
		return false; */
	if (rect.width < rect.height)
		return false;
	/*if (rank < 10)
		return false;*/
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


template<typename T> void Wavelet::print(T* arr, int n)
{
	QString res = "[";
	for (int i = 0; i < n; i++)
	{
		res += QString::number(arr[i]) + " ";
	}

	res += "]";
	qDebug() << res;
}