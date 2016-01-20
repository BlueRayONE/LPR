#ifndef LOCALISATION_HPP
#define LOCALISATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Localisation
{
public:
    Localisation();
    cv::Mat preprocess(cv::Mat& image);
private:
    cv::Mat binaryImage(cv::Mat& image);
    int thresh(cv::Mat& image, int x, int y);
};

#endif // LOCALISATION_HPP
