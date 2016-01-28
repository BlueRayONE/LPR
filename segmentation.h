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

    int* computeHorizontalHistogram(const cv::Mat& image);
    int* computeVerticalHistogram(const cv::Mat& image);
    cv::Mat cropHorizontal(const cv::Mat& image);
    cv::Mat cropImage(const cv::Mat& image);

    static void segmentationTest(const cv::Mat& testImage);
    static void writeIntoFile(int* array, int length, std::string filename);

    cv::Mat originalImage;

private:
    cv::Mat computeBinaryImage(const cv::Mat& image);
    int getVerticalStart(const cv::Mat& image);
    int getVerticalEnd(const cv::Mat& image);
    int getHorizontalStart(const cv::Mat& image);
    int getHorizontalEnd(const cv::Mat& image);

    cv::Mat croppedBinaryImage;
};

#endif // SEGMENTATION_HPP
