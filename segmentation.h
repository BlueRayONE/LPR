#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


class Segmentation
{
public:
    Segmentation(const cv::Mat& image);
    ~Segmentation();

    int* computeHorizontalHistogram();
    int* computeVerticalHistogram();
    cv::Mat cropHorizontal();
    cv::Mat cropVertical();

    cv::Mat originalImage;

private:
    cv::Mat computeBinaryImage();
    cv::Mat binaryImage;

    int getHorizontalStart(int* horizontalHistogram);
    int getHorizontalEnd(int* horizontalHistogram);
};

#endif // SEGMENTATION_HPP
