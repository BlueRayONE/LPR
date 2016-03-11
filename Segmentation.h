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

    cv::Mat cropImage(const cv::Mat& binaryImage);
    bool findChars();

    cv::Mat originalImage;
    // cropImage must be executed before to initialize croppedImage, croppedBinaryImage
    cv::Mat croppedBinaryImage;
    cv::Mat croppedImage;

    // findChars must be executed before to initialize chars
    std::vector<cv::Mat> chars; //LP hat max. 9 Zeichen: WAF-MU 3103 (+1 Puffer)

private:

    // Preprocessing
    int getVerticalStart(const cv::Mat& image);
    int getVerticalEnd(const cv::Mat& image);
    int getHorizontalStart(const cv::Mat& image);
    int getHorizontalEnd(const cv::Mat& image);
    double computeAngle(const cv::Mat& image);
    double computeShearingAngle(const cv::Mat& blackened);
    cv::Mat rotate(const cv::Mat& toRotate);
    cv::Mat equalizeImage(const cv::Mat& image);
    cv::Mat shear(const cv::Mat& blackened);
    void blackenEuroline(cv::Mat& horizontalCropped);
    cv::Mat cropHorizontal(const cv::Mat& binaryImage);
    cv::Mat computeBinaryImage(cv::Mat image, NiblackVersion version, int windowSize);

    // Segmentation
    bool isBadge(const cv::Mat& imageSegment);
    bool isBadgeDetail(const cv::Mat& imageSegment, bool reverse);
    int findChange(int* colProjection, int start, int maxPos);
    int findValley(int* colProjection, int size, int position, int thresholdValley);
    int findPeak(int *colProjection, int size, int position, int thresholdPeak);

    // Utils
    int* reverseArray(int *arr, int start, int end);
    bool isInInterval(int value, std::pair<int,int> interval);
    void plotArray(int* array, int length, std::string filename, bool rm, bool view);
    int* computeColProjection(const cv::Mat& image, NiblackVersion version);
    int* computeRowProjection(const cv::Mat& image);

};

#endif // SEGMENTATION_HPP
