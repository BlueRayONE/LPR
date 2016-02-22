#include "segmentation.h"
#include "ImageViewer.h"
#include "binarizewolfjolion.h"

#include <iostream>
#include <fstream>
#include <string>
//#include <tesseract/ocrclass.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

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

    for(int charNo=0; charNo<10; charNo++){
        if(rightPos >= size) break;

        leftPos = rightPos; // End of prev elem is start of new elem
        tmpPos=findPeak(horizontalHistogram,size,leftPos); //Peak des nächsten Elements finden

        if(tmpPos == -1){ //Keinen Peak gefunden!!
            failed = true;
            break;
        }
        else if(tmpPos == -2) //Ende erreicht
        {

        }

        rightPos=findValley(horizontalHistogram,size,tmpPos) + 2; //Ende des nächsten Elements finden und bisschen was drauf rechnen

        if(rightPos == -1){ //Keinen Valley gefunden!!
            failed = true;
            break;
        }
        else if(rightPos == -2) //Ende erreicht
        {

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
    const int thresholdValley = 3; //threshold value, which should indicate beginning of a valley
    int result = -1;

    for(int i=position; i < size; i++){ //Punkt finden der Schwellwert unterschreitet
        if(horizontalHistogram[i] <= thresholdValley){
            while(horizontalHistogram[i+1] < horizontalHistogram[i] && i<size) i++; //lok. Minimum finden
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5)) result = -2; //Ende des Bildes erreicht?

    return result;
}

int Segmentation::findPeak(int *horizontalHistogram, int size, int position)
{
    const int thresholdPeak = 30; //threshold value, which should indicate beginning of a peak
    int result = -1;

    for(int i=position; i < size; i++){ //Punkt finden der Schwellwert überschreitet
        if(horizontalHistogram[i] >= thresholdPeak){ //lok. Maximum finden
            while(horizontalHistogram[i+1] > horizontalHistogram[i] && i<size) i++;
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5)) result = -2; //Ende des Bildes erreicht?

    return result;
}


double Segmentation::computeSlope(const Mat& image, bool horizontal){
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION);

    Mat dst, cdst;
    // Parameters must be optimized
    Canny(binaryImage, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;

    int minLinLength, threshold;
    if(horizontal == true){
        threshold = 100;
        minLinLength = 1/3 * cdst.cols;
    } else {
        threshold = cdst.rows * (75.0/119);
        minLinLength = 1/3 * cdst.rows;
    }
    // Parameters must be optimized
    HoughLinesP(dst, lines, 1, CV_PI/180, threshold, minLinLength, 15);

    double maxLength = 0;
    double slope = 0;
    for(size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);

        double currentLength = sqrt(pow((l[0] - l[2]), 2) + pow((l[1] - l[3]), 2));
        if(currentLength > maxLength){
            maxLength = currentLength;
            slope = (double)(l[3] - l[1]) / (l[2] - l[0]);
        }
    }

    imshow("detected lines", cdst);
    return slope;
}

Mat Segmentation::rotate(const cv::Mat& toRotate){
    double slope = computeSlope(toRotate, true);

    double angle = atan(slope) * 180/PI;
    Point2f pt(toRotate.cols/2., toRotate.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    Mat rotated = Mat(toRotate.rows, toRotate.cols, toRotate.type());
    warpAffine(toRotate, rotated, r, Size(toRotate.cols, toRotate.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
    cout << "arctan(" << slope << ") = " << angle << " degree " << endl;
    return rotated;
}

void Segmentation::segmentationTest(const cv::Mat& testImage){
    Segmentation segmentation(testImage);

    Mat croppedImage = segmentation.cropImage(testImage);
    ImageViewer::viewImage(croppedImage, "Cropped Image");

//    Mat hsvImage;
//    cvtColor(testImage, hsvImage, CV_BGR2HSV);

//    for(int i = 0; i < testImage.rows; i++){
//        for(int j = 0; j < testImage.cols; j++){
//            Vec3b pixel = hsvImage.at<Vec3b>(i,j);
//            int hue = pixel[0];
//            int saturation = pixel[1];
//            int value = pixel[2];

//            pair<int,int> h, s, v;
//            h.first = 105;
//            h.second = 120;
//            s.first = 140;
//            s.second = 255;
//            v.first = 7.0/10 *255;
//            v.second = 255;

//            if(segmentation.isInInterval(hue,h)){
//                pixel[1] = 255;
//                pixel[2] = 255;
//                hsvImage.at<Vec3b>(i,j) = pixel;
//            }
//        }
//    }
//    Mat outputImage;
//    cvtColor(hsvImage, outputImage, CV_HSV2BGR);
//    imshow("test for blue detection", outputImage);


}

Mat Segmentation::computeBinaryImage(Mat image, NiblackVersion version){
    Mat greyImage, image8bit;

//    filteredImage = Mat(image.rows, image.cols, image.type());
//    bilateralFilter(image, filteredImage, 9, 100, 1000);
    cvtColor(image, greyImage, CV_BGR2GRAY);
    greyImage.convertTo(image8bit, CV_8UC1);
    Mat binaryImage(greyImage.rows, greyImage.cols, CV_8UC1);

    int window = 40;
    NiblackSauvolaWolfJolion(greyImage, binaryImage, version, window, window, 0.5, 128);
    return binaryImage;
}

Mat Segmentation::cropHorizontal(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    delete verticalHistogram;

    int start = getVerticalStart(image);
    int end = getVerticalEnd(image);

    Mat horizontalCropped = image(Rect(0,start, image.cols, end-start));
    return horizontalCropped;
}

Mat Segmentation::cropImage(const Mat& image){
    // Histogram Equalization of Color image
    vector<Mat> channels;
    Mat equalizedImage;
    cvtColor(image, equalizedImage, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(equalizedImage,channels); //split the image into channels
    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
    merge(channels,equalizedImage); //merge 3 channels including the modified 1st channel into one image
    cvtColor(equalizedImage, equalizedImage, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

    imshow("equalized", equalizedImage);

    // First, filter the image to sharpen the edges and remove noise
    Mat filteredImage = Mat(equalizedImage.rows, equalizedImage.cols, equalizedImage.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);

    Mat rotated = rotate(filteredImage);
    //imshow("Rotated", rotated);
    Mat horizontalCropped = cropHorizontal(rotated);

    // make all blue parts black
/*    Mat hsvImage;
    cvtColor(horizontalCropped, hsvImage, CV_BGR2HSV);
    MatIterator_<Vec3b> it = hsvImage.begin<Vec3b>();
    MatIterator_<Vec3b> it_end = hsvImage.end<Vec3b>();
    for(; it != it_end; ++it){
        // work with pixel in here, e.g.:
        Vec3b& pixel = *it; // reference to pixel in image
        int hue = pixel[0];
        int saturation = pixel[1];
        int value = pixel[2];

        pair<int,int> h, s, v;
        h.first = 100;
        h.second = 130;
        s.first = 4.0/10 * 255;
        s.second = 255;
        v.first = 6.0/10 *255;
        v.second = 255;

        if(isInInterval(hue,h) && isInInterval(saturation,s) && isInInterval(value,v)){
            //make it black
            pixel[2] = 0;
        }
    }
    cvtColor(hsvImage, horizontalCropped, CV_HSV2BGR);
    imshow("blue to black", horizontalCropped);*/


    //Shearing
    double slope = computeSlope(horizontalCropped, false);
    Mat warpedImage = Mat(horizontalCropped.rows, horizontalCropped.cols, horizontalCropped.type());
    Mat shearMat = (Mat_<double>(3,3) << 1, -1/slope, 0, 0, 1, 0, 0, 0, 1);
    Size size(horizontalCropped.cols, horizontalCropped.rows);
    warpPerspective(horizontalCropped, warpedImage, shearMat, size);
    //imshow("Sheared", warpedImage);

    imshow("wo ist die linie", computeBinaryImage(warpedImage, WOLFJOLION));

    int start = getHorizontalStart(warpedImage);
    int end = getHorizontalEnd(warpedImage);
    Mat croppedImage = warpedImage(Rect(start, 0, end-start, horizontalCropped.rows));
    croppedBinaryImage = computeBinaryImage(croppedImage, WOLFJOLION);
    imshow("WOLF", croppedBinaryImage);

    return croppedImage;
}


bool Segmentation::isInInterval(int value, std::pair<int,int> interval){
    return ((value >= interval.first) && (value <= interval.second));
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
    int* horizontalHistogram = computeHorizontalHistogram(image, WOLFJOLION);
    int width = image.cols;
    int middleRow = width/2;
    writeIntoFile(horizontalHistogram, image.cols, "Horizontal.txt");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Horizontal.txt' with linespoint\"");

    int maxValue = 0;
    int indexAtMax = 0;
    // start from the middle row and search till the first col
    // find index with maximum value
    for(int i = middleRow; i >= 0; i--){
        int currentValue = horizontalHistogram[i];

        if(currentValue > maxValue){
            maxValue = currentValue;
            indexAtMax = i;
        }
    }
    delete horizontalHistogram;
    cout << maxValue << " at index " << indexAtMax << endl;
    return indexAtMax + 10;
}

int Segmentation::getHorizontalEnd(const Mat& image){
    int* horizontalHistogram = computeHorizontalHistogram(image, WOLFJOLION);
    int width = image.cols;
    int middleRow = width/2;
    writeIntoFile(horizontalHistogram, image.cols, "Horizontal.txt");

    int maxValue = 0;
    int indexAtMax = 0;
    //start from the middle row and search till the end col
    // find index with maximum value
    for(int i = middleRow; i < width; i++){
        int currentValue = horizontalHistogram[i];

        if(currentValue > maxValue){
            maxValue = currentValue;
            indexAtMax = i;
        }
    }
    delete horizontalHistogram;
    return indexAtMax - 10;
}

int Segmentation::getVerticalStart(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    //very important: don't mix cols with rows -> bad results
    writeIntoFile(verticalHistogram, image.rows, "Vertical.txt");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");

    int offset = 1;
    int borderThickness = 5;
    int height = image.rows;
    int middle = height/2;
    int startIndex = middle;

    int sum = 0;
    int interval = height*0.15;
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
    int borderThickness = 5;
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

int* Segmentation::computeHorizontalHistogram(const Mat& image, NiblackVersion version){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, version);

    int* histogram = new int[width];
    for(int i = 0; i < width; i++){
       histogram[i] = height - countNonZero(binaryImage.col(i));
    }
    return histogram;
}

int* Segmentation::computeVerticalHistogram(const Mat& image){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION);
    //imshow("wolfilein", binaryImage);

    int* histogram = new int[height];
    for(int i = 0; i < height; i++){
       histogram[i] = width - countNonZero(binaryImage.row(i));
    }
    return histogram;
}
