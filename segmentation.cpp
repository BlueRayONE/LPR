#include "segmentation.h"
#include "ImageViewer.h"
#include "binarizewolfjolion.h"

#include <iostream>
#include <fstream>
#include <string>
//#include <tesseract/ocrclass.h>

#define PI 3.14159265
#define WINDOW_SIZE 45

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
    int i;

    for(i=position; i < size; i++){ //Punkt finden der Schwellwert unterschreitet
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
    int i;

    for(i=position; i < size; i++){ //Punkt finden der Schwellwert überschreitet
        if(horizontalHistogram[i] >= thresholdPeak){ //lok. Maximum finden
            while(horizontalHistogram[i+1] > horizontalHistogram[i] && i<size) i++;
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5)) result = -2; //Ende des Bildes erreicht?

    return result;
}

int Segmentation::slopeBetweenPoints(pair<int,int> p0, pair<int,int> p1){
    return (p1.second - p0.second) / (p1.first - p0.first);
}

double Segmentation::computeAngle(const Mat& image, bool horizontal){
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION, WINDOW_SIZE);

    Mat dst, cdst;
    // Parameters must be optimized
    Canny(binaryImage, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;

    int minLinLength, threshold;
    if(horizontal){
        threshold = 100;
        minLinLength = 1/2 * cdst.cols;
    } else {
        threshold = cdst.rows * 0.5;
        minLinLength = 1/3 * cdst.rows;
    }
    // Parameters must be optimized
    HoughLinesP(dst, lines, 1, CV_PI/180, threshold, minLinLength, 20);

    int lineIndex = 0;
    double angle = 0;
    double maxLength = 0;
    bool found = false;

    if(horizontal){
        // default angle
        angle = 0;
        for(size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);

            double currentSlope = (double)(l[3] - l[1]) / (l[2] - l[0]);
            double currentAngle = atan(currentSlope) * 180/PI;
            double currentLength = sqrt(pow((l[0] - l[2]), 2) + pow((l[1] - l[3]), 2));

            if((currentLength > maxLength) && (abs(currentAngle) <= 45)){
                maxLength = currentLength;
                lineIndex = i;
                angle = currentAngle;
                found = true;
            }
        }
        } else {
        // default angle
        angle = 90;
        for(size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);

            double currentSlope = (double)(l[3] - l[1]) / (l[2] - l[0]);
            double currentAngle = atan(currentSlope) * 180/PI;
            double currentLength = sqrt(pow((l[0] - l[2]), 2) + pow((l[1] - l[3]), 2));

            if((currentLength > maxLength) && (abs(currentAngle) > 45)){
                maxLength = currentLength;
                lineIndex = i;
                angle = currentAngle;
                found = true;
            }

        }
    }


    if(found){
        Vec4i l = lines[lineIndex];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
    }

    if(horizontal)
        imshow("Detect horizontal lines", cdst);
    else
        imshow("Detect vertical lines", cdst);

    return angle;
}

Mat Segmentation::rotate(const cv::Mat& toRotate){
    double angle = computeAngle(toRotate, true);

    Point2f pt(toRotate.cols/2., toRotate.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    Mat rotated = Mat(toRotate.rows, toRotate.cols, toRotate.type());
    warpAffine(toRotate, rotated, r, Size(toRotate.cols, toRotate.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
    cout << "angle: " << angle << endl;
    return rotated;
}

void Segmentation::segmentationTest(const cv::Mat& testImage){
    Segmentation segmentation(testImage);

    Mat croppedImage = segmentation.cropImage(testImage);
    ImageViewer::viewImage(croppedImage, "Cropped Image");
    //imshow("my binary", segmentation.computeBinaryImage(testImage, WOLFJOLION));
}

Mat Segmentation::computeBinaryImage(Mat image, NiblackVersion version, int windowSize){
    Mat greyImage;

    cvtColor(image, greyImage, CV_BGR2GRAY);
    Mat binaryImage(greyImage.rows, greyImage.cols, CV_8UC1);

    NiblackSauvolaWolfJolion(greyImage, binaryImage, version, windowSize, windowSize, 0.5, 128);
    return binaryImage;
}

Mat Segmentation::cropHorizontal(const Mat& image){
    int start = getVerticalStart(image);
    int end = getVerticalEnd(image);

    if(start < end){
        Mat horizontalCropped = image(Rect(0, start, image.cols, end-start));
        return horizontalCropped;
    } else
        return image;
}


Mat Segmentation::equalizeImage(const Mat& image){
    vector<Mat> channels;
    Mat equalizedImage;
    cvtColor(image, equalizedImage, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(equalizedImage,channels); //split the image into channels
    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
    merge(channels,equalizedImage); //merge 3 channels including the modified 1st channel into one image
    cvtColor(equalizedImage, equalizedImage, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

    return equalizedImage;
}

Mat Segmentation::shear(const Mat& image, double slope){
    Mat warpedImage = Mat(image.rows, image.cols, image.type());
    Mat shearMat = (Mat_<double>(3,3) << 1, -1/slope, 0, 0, 1, 0, 0, 0, 1);
    Size size(image.cols, image.rows);
    warpPerspective(image, warpedImage, shearMat, size);

    return warpedImage;
}

Mat Segmentation::cropImage(const Mat& image){
    // Histogram Equalization of Color image
    Mat equalizedImage = equalizeImage(image);
    cout << "Nach equalization" << endl;

    // Filter the image to sharpen the edges and remove noise
    Mat filteredImage = Mat(equalizedImage.rows, equalizedImage.cols, equalizedImage.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);
    cout << "Nach dem Filtern" << endl;

    // Rotate the image
    Mat rotated = rotate(filteredImage);
    imshow("Rotiert", rotated);
    cout << "Nach dem Rotieren" << endl;
    imshow("Nach dem Rotieren", computeBinaryImage(rotated, WOLFJOLION, WINDOW_SIZE));


    // Crop horizontal
    Mat horizontalCropped = cropHorizontal(rotated);
    cout << "Nach dem horizontal cropping" << endl;
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Horizontal.txt' with linespoint\"");


    if(horizontalCropped.rows != 0 && horizontalCropped.cols != 0){
        imshow("Rotated horizontal cropped", horizontalCropped);

        // make all blue parts black
        Mat hsvImage;
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
            h.first = 90;
            h.second = 130;
            s.first = 40/100.0 * 255;
            s.second = 255;
            v.first = 50/100.0 * 255;
            v.second = 255;

            if(isInInterval(hue,h) && isInInterval(saturation,s) && isInInterval(value,v)){
                //make it black
                pixel[2] = 0;
            }
        }
        cvtColor(hsvImage, horizontalCropped, CV_HSV2BGR);
        imshow("blue to black", horizontalCropped);
        cout << "Nach dem Schwarzmachen" << endl;


        double angle = computeAngle(horizontalCropped, false);
        double slope = tan(angle*PI/180);
        Mat sheared = shear(horizontalCropped, slope);
        imshow("Sheared", sheared);
        cout << "Nach dem Scheren" << endl;


        // Crop at the left and right side
        int start = getHorizontalStart(sheared);
        int end = getHorizontalEnd(sheared);
        if(start < end){
            Mat croppedImage = sheared(Rect(start, 0, end-start, horizontalCropped.rows));
            croppedBinaryImage = computeBinaryImage(croppedImage, WOLFJOLION, 70);
            imshow("Cropped binary image", croppedBinaryImage);

            cout << "Nach dem gesamten Cropping" << endl;
            return croppedImage;
        } else {
            croppedBinaryImage = computeBinaryImage(sheared, WOLFJOLION, 60);
            imshow("Cropped binary image", croppedBinaryImage);

            return sheared;
        }

    } else {
        cerr << "Horizontal cropping failed" << endl;
    }

    return image;
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
    return indexAtMax + 5;
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
    return indexAtMax - 5;
}

int Segmentation::getVerticalStart(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    //very important: don't mix cols with rows -> bad results
    writeIntoFile(verticalHistogram, image.rows, "Vertical.txt");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");

    int height = image.rows;
    int middle = height/2;
    int startIndex = middle;

    // must be dynamic (7 seems to be a good factor)
    int threshold = image.cols / 7;

    // start from the middle row and search till the first row
    int maximum = verticalHistogram[startIndex];
    bool found = false;
    int i = 1;
    while(!found){
        int currentValue = verticalHistogram[startIndex - i];

        if(currentValue < maximum){
            int diff = maximum - currentValue;
            if(diff >= threshold){
                found = true;
                int index = startIndex - i;
                int currentMin = verticalHistogram[index];
                int neighborMin = verticalHistogram[index - 1];
                while(neighborMin < currentMin){
                    index = index - 1;
                    currentMin = verticalHistogram[index];
                    neighborMin = verticalHistogram[index - 1];
                }
                startIndex = index;
            }
            i++;
        } else {
            maximum = currentValue;
            startIndex = startIndex - i;
            i = 1;
        }
    }

    delete verticalHistogram;
    if(isInInterval(startIndex, pair<int,int>(0, image.rows-1)))
        return startIndex;
    else
        return 0;
}

int Segmentation::getVerticalEnd(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    //very important: don't mix cols with rows -> bad results
    writeIntoFile(verticalHistogram, image.rows, "Vertical.txt");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");

    int height = image.rows;
    int middle = height/2;
    int endIndex = middle;

    // must be dynamic
    int threshold = image.cols / 7;

    // start from the middle row and search till the first row
    int maximum = verticalHistogram[endIndex];
    bool found = false;
    int i = 1;
    while(!found){
        int currentValue = verticalHistogram[endIndex + i];

        if(currentValue < maximum){
            int diff = maximum - currentValue;
            if(diff >= threshold){
                found = true;
                int index = endIndex + i;
                int currentMin = verticalHistogram[index];
                int neighborMin = verticalHistogram[index + 1];
                while(neighborMin < currentMin){
                    index = index + 1;
                    currentMin = verticalHistogram[index];
                    neighborMin = verticalHistogram[index + 1];
                }
                endIndex = index;
            }
            i++;
        } else {
            maximum = currentValue;
            endIndex = endIndex + i;
            i = 1;
        }
    }

    delete verticalHistogram;
    if(isInInterval(endIndex, pair<int,int>(0, image.rows-1)))
        return endIndex;
    else
        return 0;
}

int* Segmentation::computeHorizontalHistogram(const Mat& image, NiblackVersion version){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, version, WINDOW_SIZE);

    int* histogram = new int[width];
    for(int i = 0; i < width; i++){
       histogram[i] = height - countNonZero(binaryImage.col(i));
    }
    return histogram;
}

int* Segmentation::computeVerticalHistogram(const Mat& image){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION, WINDOW_SIZE);
    //imshow("wolfilein", binaryImage);

    int* histogram = new int[height];
    for(int i = 0; i < height; i++){
       histogram[i] = width - countNonZero(binaryImage.row(i));
    }
    return histogram;
}
