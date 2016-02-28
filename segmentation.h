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
    cv::Mat* findChars(const cv::Mat &originalImage);


    static void segmentationTest(const cv::Mat& testImage);
    cv::Mat originalImage;

    bool isBadge(const cv::Mat& imageSegment);
    int findChange(int *horizontalHistogram, int start, int maxPos);
    int findValley(int *horizontalHistogram, int size, int position, int thresholdValley);
    int findPeak(int *horizontalHistogram, int size, int position, int thresholdPeak);

    cv::Mat croppedBinaryImage;
    cv::Mat croppedImage;
    cv::Mat* chars; //LP hat max. 9 Zeichen: WAF-MU 3103 (+1 Puffer)

private:
    cv::Mat computeBinaryImage(cv::Mat image, NiblackVersion version, int windowSize);
    int getVerticalStart(const cv::Mat& image);
    int getVerticalEnd(const cv::Mat& image);
    int getHorizontalStart(const cv::Mat& image);
    int getHorizontalEnd(const cv::Mat& image);

    double computeAngle(const cv::Mat& image, bool horizontal);
    cv::Mat rotate(const cv::Mat& toRotate);
    cv::Mat equalizeImage(const cv::Mat& image);
    cv::Mat shear(const cv::Mat& image, double slope);
    int slopeBetweenPoints(std::pair<int,int> p0, std::pair<int,int> p1);

    //utils
    int* reverseArray(int *arr, int start, int end);
    bool isInInterval(int value, std::pair<int,int> interval);
    void plotArray(int* array, int length, std::string filename, bool rm, bool view);
};

#endif // SEGMENTATION_HPP
