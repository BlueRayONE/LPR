#include "segmentation.h"
#include "ImageViewer.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;

Segmentation::Segmentation(const Mat& image): originalImage(image){
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
                chars[charNo-1]= croppedBinaryImage(Rect(leftPos,0, rightPos-leftPos, croppedBinaryImage.cols)); //-1 weil wenns Plaketten waren nichts eingefügt wurde
        }else{
            if(true /*!isBadge(horizontalHistogram,leftPos,rightPos)*/){ //Es handelt sich nicht um Bereich der Plaketten
//                chars[charNo]= binaryImage(Rect(0,leftPos, binaryImage.cols, rightPos-leftPos));
                chars[charNo]= croppedBinaryImage(Rect(cv::Point(rightPos, 0), cv::Point(rightPos-leftPos, croppedBinaryImage.cols)));
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

void Segmentation::segmentationTest(const cv::Mat& testImage){
    Segmentation segmentation(testImage);

    int* horizontal = segmentation.computeHorizontalHistogram(testImage);
    int* vertical = segmentation.computeVerticalHistogram(testImage);

    //very important: don't mix cols with rows -> bad results
    writeIntoFile(horizontal, testImage.cols, "Horizontal.txt");
    writeIntoFile(vertical, testImage.rows, "Vertical.txt");

    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Horizontal.txt' with linespoint\"");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");

    delete horizontal;
    delete vertical;

    ImageViewer::viewImage(segmentation.cropImage(testImage), "Cropped Image");
}

int* Segmentation::computeHorizontalHistogram(const Mat& image){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image);

    int* histogram = new int[width];
    for(int i = 0; i < width; i++){
       histogram[i] = height - countNonZero(binaryImage.col(i));
    }

    return histogram;
}

int* Segmentation::computeVerticalHistogram(const Mat& image){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image);

    int* histogram = new int[height];
    for(int i = 0; i < height; i++){
       histogram[i] = width - countNonZero(binaryImage.row(i));
    }

    return histogram;
}

Mat Segmentation::computeBinaryImage(const Mat& image){
    Mat filteredImage, greyImage, image8bit;

    filteredImage = Mat(image.rows, image.cols, image.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);

    cvtColor(filteredImage, greyImage, CV_BGR2GRAY);
    greyImage.convertTo(image8bit, CV_8UC1);
    Mat binaryImage = Mat(greyImage.rows, greyImage.cols, greyImage.type());

    // threshold must be set manual
    threshold(image8bit, binaryImage, 100, 255, CV_THRESH_BINARY);

    return binaryImage;
}

Mat Segmentation::cropHorizontal(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    writeIntoFile(verticalHistogram, image.rows, "Vertical.txt");
    delete verticalHistogram;

    int start = getVerticalStart(image);
    int end = getVerticalEnd(image);

    Mat horizontalCropped = image(Rect(0,start, image.cols, end-start));
    return horizontalCropped;
}

Mat Segmentation::cropImage(const Mat& image){
    Mat horizontalCropped = cropHorizontal(image);

//    int start = getHorizontalStart(horizontalCropped);
//    int end = getHorizontalEnd(horizontalCropped);

//    Mat croppedImage = horizontalCropped(Rect(start, 0, end-start, horizontalCropped.rows));
//    croppedBinaryImage = croppedImage;

    return horizontalCropped;
}

void Segmentation::writeIntoFile(int* array, int length, string filename){
    std::ofstream myfile;
    myfile.open(filename);
    std::stringstream ss;

    for(int i = 0; i < length; i++){
        ss << array[i];
        ss << "\n";
    }
    myfile << ss.str();
    myfile.close();
}

int Segmentation::getHorizontalStart(const Mat& image){
    int offset = 20;
    int borderThickness = 3;

    int* horizontalHistogram = computeHorizontalHistogram(image);
    int length = image.cols;
    int middle = length/2;
    int startIndex = middle;

    int sum = 0;
    for(int i = -40; i < 40; i++){
        sum += horizontalHistogram[middle+i];
    }
    int threshold = 5;

    // start from the middle row and search till the first col
    for(int i = length/2; i >= 0; i--){
        int current = horizontalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isStart = true;
            for(int j = candidate - 1; j >= candidate - offset; j--){

                if(horizontalHistogram[j] > threshold){
                    isStart = false;
                }
            }
            if(isStart){
                startIndex = candidate - borderThickness;
                return startIndex;
            }
        }
    }
    delete horizontalHistogram;
    return startIndex;
}

int Segmentation::getHorizontalEnd(const Mat& image){
    int offset = 24;
    int borderThickness = 3;

    int* horizontalHistogram = computeHorizontalHistogram(image);
    int length = image.cols;
    int middle = length/2;
    int endIndex = middle;

    int sum = 0;
    for(int i = -40; i < 40; i++){
        sum += horizontalHistogram[middle+i];
    }
    int threshold = 5;

    //start from the middle row and seach till the end col
    for(int i = length/2; i < length; i++){
        int current = horizontalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isEnd = true;

            for(int j = candidate + 1; j < candidate + offset; j++){
                if(horizontalHistogram[j] > threshold){
                    isEnd = false;
                }
            }
            if(isEnd){
                endIndex = candidate + borderThickness;
                return endIndex;
            }

        }
    }
    delete horizontalHistogram;
    return endIndex;
}

int Segmentation::getVerticalStart(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    //very important: don't mix cols with rows -> bad results
    writeIntoFile(verticalHistogram, image.rows, "Vertical.txt");
    system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");

    int offset = 1;
    int borderThickness = 1;
    int length = image.rows;
    int middle = length/2;
    int startIndex = middle;

    int sum = 0;
    int interval = length*0.15;
    for(int i = -interval; i < interval; i++){
        sum += verticalHistogram[middle+i];
    }
    int threshold = sum/(interval*2) * 0.5;

    // start from the middle row and search till the first row
    for(int i = middle; i >= 0; i--){
        int current = verticalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isStart = true;
            for(int j = candidate - 1; j >= candidate - offset; j--){

                if(verticalHistogram[j] > threshold){
                    isStart = false;
                }
            }
            if(isStart){
                startIndex = candidate - borderThickness;
                return startIndex;
            }
        }
    }
    delete verticalHistogram;
    return startIndex;
}

int Segmentation::getVerticalEnd(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);

    int offset = 1;
    int borderThickness = 1;
    int length = image.rows;
    int middle = length/2;
    int endIndex = middle;

    int sum = 0;
    int interval = length*0.15;
    for(int i = -interval; i < interval; i++){
        sum += verticalHistogram[middle+i];
    }
    int threshold = sum/(interval*2) * 0.5;

    //start from the middle row and seach till the end row
    for(int i = middle; i < length; i++){
        int current = verticalHistogram[i];

        if(current < threshold){
            int candidate = i;
            // number of successor that have to be under the threshold
            bool isEnd = true;

            for(int j = candidate + 1; j < candidate + offset; j++){
                if(verticalHistogram[j] > threshold){
                    isEnd = false;
                }
            }
            if(isEnd){
                endIndex = candidate + borderThickness;
                return endIndex;
            }

        }
    }
    delete verticalHistogram;
    return endIndex;
}
