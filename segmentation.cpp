#include "segmentation.h"
#include <iostream>

using namespace cv;

Segmentation::Segmentation(const Mat& image){
    originalImage = image;
    binaryImage = computeBinaryImage();
}

Segmentation::~Segmentation(){

}

int* Segmentation::computeHorizontalHistogram(){
    int width = originalImage.cols;
    int height = originalImage.rows;

    int* histogram = new int[width];
    imshow("Binary Image", binaryImage);
    for(int i = 0; i < width; i++){
       histogram[i] = height - countNonZero(binaryImage.col(i));
    }

    return histogram;
}

int* Segmentation::computeVerticalHistogram(){
    int width = originalImage.cols;
    int height = originalImage.rows;

    int* histogram = new int[height];
    imshow("Binary Image", binaryImage);
    for(int i = 0; i < height; i++){
       histogram[i] = width - countNonZero(binaryImage.row(i));
    }

    return histogram;
}

Mat Segmentation::computeBinaryImage(){
    Mat filteredImage, greyImage, image8bit, threshImage;

    filteredImage = Mat(originalImage.rows, originalImage.cols, originalImage.type());
    bilateralFilter(originalImage, filteredImage, 9, 100, 1000);

    cvtColor(filteredImage, greyImage, CV_BGR2GRAY);
    greyImage.convertTo(image8bit, CV_8UC1);
    threshImage = Mat(greyImage.rows, greyImage.cols, greyImage.type());

    // threshold must be set manual
    threshold(image8bit, threshImage, 100, 255, CV_THRESH_BINARY);

    return threshImage;
}

Mat Segmentation::cropHorizontal(){
    // offset of the horizontal start and end
    int start = getHorizontalStart(computeHorizontalHistogram());
    std::cout << start << std::endl;
    int end = 60;

    Mat croppedImage = originalImage(Rect(0,start, originalImage.cols, end-start));
    return croppedImage;
}

Mat Segmentation::cropVertical(){

}

int Segmentation::getHorizontalStart(int *horizontalHistogram){
    // find local minimum at the beginning
    int startIndex = 0;
    int length = originalImage.rows;
    int threshold = 0.25*originalImage.cols;

    for(int i = 0; i < length; i++){
        int current = horizontalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            for(int j = candidate + 1; j < candidate + 5; j++){
                if(horizontalHistogram[j] > threshold)
                    break;
                else {
                    startIndex = candidate + 3;
                    break;
                }
            }
            break;
        }
    }
    return startIndex;
}

int Segmentation::getHorizontalEnd(int *horizontalHistogram){

}
