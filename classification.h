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
    Classification();
    void characterRecognition(const cv::Mat& image);
};

#endif // CLASSIFICATION_HPP
