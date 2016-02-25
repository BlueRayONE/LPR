#include "ImageViewer.h"

void ImageViewer::viewImage(const cv::Mat img, std::string title, int height)
{
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    cv::imshow(title, img);
    if(height != -1)
        cv::resizeWindow(title, height*img.cols/img.rows, height);
    else
        cv::resizeWindow(title, img.cols, img.rows);

	/*int k = cv::waitKey(0);

	if (k == 's')
		cv::imwrite(title + ".jpg", img);
	else
		return;*/
}

