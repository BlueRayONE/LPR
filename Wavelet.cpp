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
    //int maxID = maxIDVal.first;
	float max = maxIDVal.second;

	double avg = std::accumulate(gauss.begin(), gauss.end(), 0.0) / (double) (binarizedHL.rows);
	avg = max;

	std::vector<std::pair<int, int>> startRowsHeights =  this->findThresholdAreas(avg, gauss);

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
	std::vector<cv::Mat> matCandidates;
#pragma region draw_candidates
	//int count = (candidatesRough.size() > 10) ? 10 : candidatesRough.size();
    size_t count = candidatesRough.size();
    for (size_t i = 0; i <  candidatesRough.size(); i++)
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

				//ONLY TEMPORARILY
				//if (i == 0) //extract first candidate
				//{
				//	test = img(cv::Rect(currentRectEx.tl().x * 2, currentRectEx.tl().y * 2, currentRectEx.width * 2, currentRectEx.height * 2));
				//}
			}

		}
			
	}

#pragma endregion
/*************************************************************************************/


	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorHL, "colorHL");
	if (DEBUG_LEVEL == 2) ImageViewer::viewImage(colorBinHL, "colorBinHL");
	//if (DEBUG_LEVEL >= 1) ImageViewer::viewImage(colorMorphHL, "colorMorphHL");
	if (DEBUG_LEVEL >= 0) ImageViewer::viewImage(img, "orig width candidates");
	//if (DEBUG_LEVEL >= 0) ImageViewer::viewImage(test, "candidate");


	//cv::destroyAllWindows();
    /*int i = 0;
	for (auto it = matCandidates.begin(); it != matCandidates.end(); ++it)
	{
		cv::Mat src = (*it);

		
		cv::Mat SRC = src.clone();

		int origRows = src.rows;
		cv::Mat colVec = SRC.reshape(1, src.rows*src.cols); // change to a Nx3 column vector
		cv::Mat colVecD, bestLabels, centers;
		int attempts = 3;
		int clusts = 5;
		double eps = 0.001;
		colVec.convertTo(colVecD, CV_32FC3, 1.0 / 255.0); // convert to floating point
		double compactness = kmeans(colVecD, clusts, bestLabels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, attempts, eps), attempts, cv::KMEANS_PP_CENTERS, centers);

		
		std::vector<std::pair<cv::Vec3b, uint>> dominantColors = std::vector<std::pair<cv::Vec3b, uint>>(clusts);
		for (int i = 0; i < clusts; i++)
		{
			dominantColors[i] = std::make_pair(cv::Vec3b(centers.at<float>(i, 0) * 255, centers.at<float>(i, 1) * 255, centers.at<float>(i, 2) * 255), 0);
		}


		//repaint image with dominant colors
		cv::Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels
		cv::Mat repaint = cv::Mat(src.rows, src.cols, src.type());


		for (int r = 0; r < src.rows; r++)
		{
			for (int c = 0; c < src.cols; c++)
			{
				uint label = labelsImg.at<uint>(r, c);
				dominantColors[label].second++;
				float blue, green, red;
				blue = centers.at<float>(label, 0);
				green = centers.at<float>(label, 1);
				red = centers.at<float>(label, 2);
				repaint.at<cv::Vec3b>(r, c) = cv::Vec3b(blue * 255, green * 255, red * 255); //label in labelsImg.at<uchar>(r,c)

			}
		}

		std::sort(dominantColors.begin(), dominantColors.end(), [&](std::pair<cv::Vec3b, uint> i1, std::pair<cv::Vec3b, uint> i2)
		{
			return i1.second >= i2.second;

		});
		//ImageViewer::viewImage(repaint, "repaint candidate");
		

		bool hasGrey = false;
		//eval dominant colors
		size_t i = 0;
		for (auto color : dominantColors)
		{
			int max = std::max({color.first[0], color.first[1], color.first[2]});
			int min = std::min({color.first[0], color.first[1], color.first[2]});
			//check if color is grey
			if (max - min <= 10)
			{
				hasGrey = true;
			}
			if (i == 2)
				break;
			i++;
		}

		if (hasGrey)  ImageViewer::viewImage(src, "candidate real");
		
		i++;
		cv::waitKey(0);

    }*/
	
}


std::vector<float> Wavelet::gaussFilter(std::vector<int> arr)
{
	//float* res = new float[n];
	std::vector<float> res = std::vector<float>(arr.size(), 0.0);
	
	/*res[0] = 0; res[1] = 0; res[2] = 0; res[3] = 0;
	res[n-1] = 0; res[n-2] = 0; res[n-3] = 0; res[n-4] = 0;*/
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

std::vector<float> Wavelet::movingAvg(std::vector<float> arr)
{
	std::vector<float> sums = std::vector<float>(arr.size() + 1, 0.0);
	std::vector<float> res = std::vector<float>(arr.size(), 0.0);

	float sum = 0;
	
	for (size_t i = 1; i < sums.size(); i++)
	{
		sum += arr[i - 1];
		sums[i] = sum;
	}

	for (size_t i = 3; i < res.size() - 1; i++)
	{
		res[i - 1] = (sums[i + 2] - sums[i - 3]) / 5;
	}

	return res;
}

std::vector<int> Wavelet::calcRowSums(cv::Mat img)
{
	//int* sums = new int[img.rows];
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

	/*for (int r = 0; r < img.rows; r++)
	{
		int currSum = 0;
		for (int c = 0; c < img.cols; c++)
		{
			currSum += img.at<uchar>(r, c);
		}

		sums[r] = currSum;
	}*/

	return sums;
}

std::vector<int> Wavelet::calcColSums(cv::Mat img)
{
	/*int* sums = new int[img.cols];
	for (int c = 0; c < img.cols; c++)
	{
		int currSum = 0;
		for (int r = 0; r < img.rows; r++)
		{
			currSum += img.at<uchar>(r, c);
		}

		sums[c] = currSum;
	}*/

	std::vector<int> sums = std::vector<int>(img.cols, 0);

	//sums[i] = 
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

std::pair<int, float> Wavelet::findMaxPeak(std::vector<std::pair<int, float>> peaks, std::vector<float> arr)
{
	float max = 0;
	if (peaks.size() > 0)
	{
		//max = arr[peakIDs[0]];
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

std::vector<std::pair<int, int>> Wavelet::findThresholdAreas(double avg, std::vector<float> rowSums, bool splitAreas)
{
	std::vector<std::pair<int, int>> startRowsHeights;

	const int maxRectHeight = (int) (MAX_RECT_HEIGHT_RATIO * rowSums.size());
	float currentWeight = AVG_WEIGHT;

	int currentHeight = 0;
	int currentRectStart = 0;
	bool wasInThresh = false;
	bool isInThresh = false;


	for (size_t r = 0; r < rowSums.size(); r++)
	{
		if (rowSums[r] >= currentWeight*avg)
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
			currentWeight = AVG_WEIGHT; //reset currentWeight for next retangles;
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

		int count = std::count(gaussColsSums.begin(), gaussColsSums.end(), 0.0);

		std::vector<std::pair<int, float>> peaks = this->findPeaks(gaussColsSums);
		std::pair<int, float> maxIDVal = this->findMaxPeak(peaks, gaussColsSums);
        //int maxID = maxIDVal.first;
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
	if (rect.width <= rect.height * 1.5) //doesnt work for usa license plates
		return false;
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


cv::Mat Wavelet::gbrHist(cv::Mat img)
{
	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split(img, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = {0, 256};
	const float* histRange = {range};

	bool uniform = true; bool accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double) hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	/*namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);*/
	return histImage;

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
