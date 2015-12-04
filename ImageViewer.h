#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class ImageViewer
{
public:
    static void viewImage(const cv::Mat img, std::string title, int height = -1);
};

#endif // IMAGEVIEWER_H
