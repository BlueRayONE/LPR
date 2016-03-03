#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "segmentation.h"
#include "Segmentation_MSER.h"


class Classification
{
public:
    Classification(const cv::Mat& image, std::string filename);
    std::vector<std::string> characterRecognition(const std::vector<cv::Mat> plates);

    cv::Mat originalImage;
    std::string filename;
};

#endif // CLASSIFICATION_HPP
