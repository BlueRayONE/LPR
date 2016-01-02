#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


class Segmentation
{
public:
    Segmentation();
    ~Segmentation();

    int* computeHorizontalHistogram(cv::Mat& image);
    int* computeVerticalHistogram(cv::Mat& image);
    cv::Mat cropHorizontal(cv::Mat& image);
    cv::Mat cropVertical(cv::Mat& image);

private:
    cv::Mat computeBinaryImage(cv::Mat& image);

    int getHorizontalStart(int* horizontalHistogram, int length);
    int getHorizontalEnd(int* horizontalHistogram, int length);
};

#endif // SEGMENTATION_HPP
