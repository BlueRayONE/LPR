#include "segmentation.h"
#include <iostream>

/*using namespace cv;

Segmentation::Segmentation(const Mat& image): originalImage(image){
    binaryImage = computeBinaryImage();

}

Segmentation::~Segmentation(){

}

Mat* Segmentation::findChars(int *horizontalHistogram, int size)
{
    Mat* chars = new Mat[10]; //LP hat max. 9 Zeichen: WAF-MU 3103 (+1 Puffer)
    int leftPos  = 0;
    int rightPos = 0;
    int tmpPos = 0;
    bool badgeFound = false;
    bool failed = false;

    rightPos = findValley(horizontalHistogram, size, 0); //blaues Euroband überspringen
    line(originalImage, cv::Point(rightPos, 0), Point(rightPos, originalImage.rows), Scalar(255, 0, 0), 1, CV_AA);

    for(int charNo=0; charNo<10; charNo++){
        if(rightPos >= size) break;

        leftPos = rightPos; // End of prev elem is start of new elem
        tmpPos=findPeak(horizontalHistogram,size,leftPos); //Peak des nächsten Elements finden

        if(tmpPos == -1){ //Keinen Peak gefunden!!
            failed = true;
            break;
        }

        rightPos=findValley(horizontalHistogram,size,tmpPos) + 2; //Ende des nächsten Elements finden und bisschen was drauf rechnen

        if(rightPos == -1){ //Kein Valley gefunden!!
            failed = true;
            break;
        }

        if(false /*badgeFound*/){
                chars[charNo-1]= binaryImage(Rect(leftPos,0, rightPos-leftPos, binaryImage.cols)); //-1 weil wenns Plaketten waren nichts eingefügt wurde
        }else{
            if(true /*!isBadge(horizontalHistogram,leftPos,rightPos)*/){ //Es handelt sich nicht um Bereich der Plaketten
//                chars[charNo]= binaryImage(Rect(0,leftPos, binaryImage.cols, rightPos-leftPos));
                chars[charNo]= binaryImage(Rect(cv::Point(rightPos, 0), cv::Point(rightPos-leftPos, binaryImage.cols)));
                line(originalImage, cv::Point(rightPos, 0), Point(rightPos, originalImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende des Buchstabens einzeichnen
            }else {
                badgeFound = true;
            }
        }

    }
    return chars;
}

bool Segmentation::isBadge(int *horizontalHistogram, int leftPos, int rightPos)
{
    int peak;
    Mat ch1, ch2, ch3;
    vector<Mat> channels(3);
    int valCh1, valCh2, valCh3;

    split(originalImage, channels);     // split img
    // get the channels (BGR order!)
    ch1 = channels[0];
    ch2 = channels[1];
    ch3 = channels[2];

    peak = findPeak(horizontalHistogram, rightPos,leftPos);
    imshow("channel image blau", ch1);
    imshow("channel image grün", ch2);
    imshow("channel image rot", ch3);

    valCh1 = originalImage.rows - countNonZero(ch1.col(peak));
    valCh2 = originalImage.rows - countNonZero(ch2.col(peak));
    valCh3 = originalImage.rows - countNonZero(ch3.col(peak));

    if(valCh1 >= 5 || valCh2 >= 5 || valCh3 >= 5){
        line(originalImage, cv::Point(peak, 0), Point(peak, originalImage.rows), Scalar(255, 0, 0), 1, CV_AA);
        imshow("image with badges", originalImage);
        return true;
    }else{
        return false;
    }
}

int Segmentation::findValley(int *horizontalHistogram, int size, int position)
{
    const int thresholdValley = 3; //threshold alue, which should indicate beginning of a valley
    int result = -1;

    for(int i=position; i < size; i++){
        if(horizontalHistogram[i] <= thresholdValley){
            while(horizontalHistogram[i+1] < horizontalHistogram[i] && i<size){
                i++;
            }
            result = i;
            break;
        }
    }

    return result;
}

int Segmentation::findPeak(int *horizontalHistogram, int size, int position)
{
    const int thresholdPeak = 30; //threshold value, which should indicate beginning of a peak
    int result = -1;

    for(int i=position; i < size; i++){
        if(horizontalHistogram[i] >= thresholdPeak){
            while(horizontalHistogram[i+1] > horizontalHistogram[i] && i<size){
                i++;
            }
            result = i;
            break;
        }
    }

    return result;
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
    int start = getVerticalStart(computeVerticalHistogram());
    int end = getVerticalEnd(computeVerticalHistogram());
    std::cout << end << std::endl;
    //int end = 60;

    Mat croppedImage = originalImage(Rect(0,start, originalImage.cols, end-start));
    return croppedImage;
}

Mat Segmentation::cropVertical(){
    return originalImage;
}

int Segmentation::getVerticalStart(int *horizontalHistogram){
    int startIndex = 0;
    int length = originalImage.rows;
    int threshold = 0.25*originalImage.cols;

    // start from the middle row and search till the first row
    for(int i = length/2; i >= 0; i--){
        int current = horizontalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isStart = true;
            for(int j = candidate - 1; j < candidate - 3; j++){

                if(horizontalHistogram[j] > threshold){
                    isStart = false;
                }
            }
            if(isStart){
                startIndex = candidate - 3;
                return startIndex;
            }
        }
    }
    return startIndex;
}

int Segmentation::getVerticalEnd(int *horizontalHistogram){
    int length = originalImage.rows;
    int endIndex = length - 1;
    int threshold = 0.25*originalImage.cols;

    //start from the middle row and seach till the end row
    for(int i = length/2; i < length; i++){
        int current = horizontalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isEnd = true;

            for(int j = candidate + 1; j > candidate + 3; j++){
                if(horizontalHistogram[j] > threshold){
                    isEnd = false;
                }
            }
            if(isEnd){
                endIndex = candidate + 3;
                return endIndex;
            }

        }
    }
    return endIndex;
}*/
