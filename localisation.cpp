#include "localisation.h"

using namespace cv;

Localisation::Localisation()
{

}

Mat Localisation::preprocess(Mat& image){
    Mat filteredImage, greyImage, image8bit, threshImage;

    filteredImage = Mat(image.rows, image.cols, image.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);

    cvtColor(filteredImage, greyImage, CV_BGR2GRAY);
    greyImage.convertTo(image8bit, CV_8UC1);

    imwrite("filtered.jpg", filteredImage);
    imwrite("grey filtered.jpg", greyImage);

    int neighborhood = 2;

    threshImage = Mat(greyImage.rows, greyImage.cols, greyImage.type());
    for(int i = neighborhood; i < greyImage.rows-neighborhood; i++){
        for(int j = neighborhood; j < greyImage.cols-neighborhood; j++){
            int diff = abs(thresh(greyImage, i, j) - greyImage.at<uchar>(i,j));
            if(diff < 30){
                threshImage.at<uchar>(i,j) = 255;
            }
            else{
                threshImage.at<uchar>(i,j) = 0;
            }
        }
    }

    return threshImage;
}



Mat Localisation::binaryImage(Mat& image){


}

int Localisation::thresh(Mat& image, int x, int y){
    int neighborhood = 2;
    int sum = 0;
    int counter = 0;
    int c = 10;

    for(int i = x-neighborhood; i < x+neighborhood; i++){
        for(int j = y-neighborhood; j < y+neighborhood; j++){
            sum += image.at<uchar>(i,j);
            counter++;
        }
    }
    int thresh = (sum - image.at<uchar>(x,y)) / counter-1;
    return thresh - c;
}














