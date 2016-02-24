#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "binarizewolfjolion.h"


class Segmentation
{
public:
    Segmentation(const cv::Mat& image);
    ~Segmentation();

    int* computeHorizontalHistogram(const cv::Mat& image, NiblackVersion version);
    int* computeVerticalHistogram(const cv::Mat& image);
    cv::Mat cropHorizontal(const cv::Mat& binaryImage);
    cv::Mat cropImage(const cv::Mat& binaryImage);
    cv::Mat* findChars(int *horizontalHistogram, int size);


    static void segmentationTest(const cv::Mat& testImage);
    static void writeIntoFile(int* array, int length, std::string filename);

    cv::Mat originalImage;

    bool isBadge(int *horizontalHistogram, int leftPos, int rightPos);
    int findValley(int *horizontalHistogram, int size, int position);
    int findPeak(int *horizontalHistogram, int size, int position);

private:
    cv::Mat computeBinaryImage(cv::Mat image, NiblackVersion version, int windowSize);
    int getVerticalStart(const cv::Mat& image);
    int getVerticalEnd(const cv::Mat& image);
    int getHorizontalStart(const cv::Mat& image);
    int getHorizontalEnd(const cv::Mat& image);

    double computeSlope(const cv::Mat& image, bool horizontal);
    cv::Mat rotate(const cv::Mat& toRotate);
    bool isInInterval(int value, std::pair<int,int> interval);
    cv::Mat equalizeImage(const cv::Mat& image);
    cv::Mat shear(const cv::Mat& image, double slope);

    cv::Mat croppedBinaryImage;
    cv::Mat croppedImage;
};

#endif // SEGMENTATION_HPP
