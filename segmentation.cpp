#include "segmentation.h"
#include <iostream>

using namespace cv;

Segmentation::Segmentation()
{
}

Segmentation::~Segmentation(){

}

int* Segmentation::computeHorizontalHistogram(Mat& image){
    int width = image.cols;
    int height = image.rows;

    Mat binaryImage = computeBinaryImage(image);
    int* histogram = new int[width];
    imshow("Binary Image", binaryImage);
    for(int i = 0; i < width; i++){
       histogram[i] = height - countNonZero(binaryImage.col(i));
    }

    return histogram;
}

int* Segmentation::computeVerticalHistogram(Mat& image){
    int width = image.cols;
    int height = image.rows;

    Mat binaryImage = computeBinaryImage(image);
    int* histogram = new int[height];
    imshow("Binary Image", binaryImage);
    for(int i = 0; i < height; i++){
       histogram[i] = width - countNonZero(binaryImage.row(i));
    }

    return histogram;
}

Mat Segmentation::computeBinaryImage(Mat& image){
    Mat filteredImage, greyImage, image8bit, threshImage;

    filteredImage = Mat(image.rows, image.cols, image.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);

    cvtColor(filteredImage, greyImage, CV_BGR2GRAY);
    greyImage.convertTo(image8bit, CV_8UC1);
    threshImage = Mat(greyImage.rows, greyImage.cols, greyImage.type());

    // threshold must be set manual
    threshold(image8bit, threshImage, 100, 255, CV_THRESH_BINARY);

    return threshImage;
}

Mat Segmentation::cropHorizontal(Mat& image){
    // offset of the horizontal start and end
    getHorizontalStart(computeHorizontalHistogram(image), image.rows);
}

Mat Segmentation::cropVertical(Mat& image){

}

int Segmentation::getHorizontalStart(int *horizontalHistogram, int length){
    // find local minimum at the beginning
    int startIndex = 0;
    for(int i = 1; i < length; i++){
        int current = horizontalHistogram[i];
        int predecessor = horizontalHistogram[i-1];

        // !!! magic number !!! TODO
        if((current < predecessor) && (current < 40)){
            startIndex = current;
        }
    }

    std::cout << startIndex << std::endl;
    return startIndex;
}

int Segmentation::getHorizontalEnd(int *horizontalHistogram, int length){

}
