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

Segmentation::Segmentation(const Mat& image, string filename): originalImage(image), name(filename){
}

Segmentation::~Segmentation(){

}

bool Segmentation::findChars()
{
    //chars = new Mat[11]; //LP hat max. 9 Zeichen: WAF-MU 3103 (+1 Puffer)
    chars = std::vector<cv::Mat>();
    int leftPos  = 0;
    int rightPos = 0;
    int tmpPos = 0;
    bool badgeFound = false;
    bool failed = false;
    int size = croppedImage.cols;

    int* horizontalHistogram = computeHorizontalHistogram(croppedImage, WOLFJOLION);

    plotArray(horizontalHistogram, size, "horizontalFULL.txt",false,true);

    for(int charNo=0; charNo<11; charNo++){
        if(rightPos >= size-10) break;

        leftPos = rightPos; // End of prev elem is start of new elem
        tmpPos=findPeak(horizontalHistogram,size,leftPos,30); //Peak des nächsten Elements finden

        if(tmpPos == -1){ //Keinen Peak gefunden!!
            failed = true;
            break;
        }
        else if(tmpPos == -2) break;//Ende erreicht


        rightPos=findValley(horizontalHistogram,size,tmpPos,3) + 2; //Ende des nächsten Elements finden und bisschen was drauf rechnen

        if(rightPos == -1){ //Kein Valley gefunden!!
            failed = true;
            break;
        }
        else if(rightPos == -2) break; //Ende erreicht

        //Badge muss innerhalb der ersten 4 Iterationen gefunden worden sein => charNo <= 3
        if(badgeFound || charNo > 3 || !isBadge(croppedImage(Rect(leftPos, 0, rightPos-leftPos, croppedImage.rows)))){ //Es handelt sich nicht um Bereich der Plaketten
            line(croppedImage, cv::Point(rightPos, 0), Point(rightPos, croppedImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende des Buchstabens einzeichnen
            tmpPos = findChange(horizontalHistogram,leftPos+2,rightPos);
            if((tmpPos-leftPos) > 30 && !badgeFound && charNo <= 3){ //ist Plakette bei Binärisierung gefiltert worden? Dann gibts ne große weiße Lücke!
                badgeFound = true;
                line(croppedImage, cv::Point(tmpPos-5, 0), Point(tmpPos-5, croppedImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende des Leerraums einzeichnen
                line(croppedImage, cv::Point((leftPos+((tmpPos-leftPos)/2)), 0), cv::Point((leftPos+((tmpPos-leftPos)/2)), croppedImage.rows), Scalar(0, 0, 255), 1, CV_AA); // Badge mittig markieren
                chars.push_back(cv::Mat(1,1,CV_8U));
                charNo++; //das war nun ein "außerordentlicher" Char mehr im Vektor, also erhöhen
                leftPos = tmpPos-5; // leftPos für Anfang vom eigentlichen Buchstaben auf Ende des Leerraums setzen
            }
            chars.push_back(croppedBinaryImage(Rect(leftPos, 0, rightPos-leftPos, croppedBinaryImage.rows)));
        }else{
            badgeFound = true;
            chars.push_back(cv::Mat(1,1,CV_8U));
            line(croppedImage, cv::Point((leftPos+((rightPos-leftPos)/2)), 0), cv::Point((leftPos+((rightPos-leftPos)/2)), croppedImage.rows), Scalar(0, 0, 255), 1, CV_AA); // Badge mittig markieren
            rightPos=findValley(horizontalHistogram,size,tmpPos,3) + 2;
            line(croppedImage, cv::Point(rightPos, 0), Point(rightPos, croppedImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende der Plakette einzeichnen
        }


    }

    ImageViewer::viewImage(croppedImage, "segmented", 400);
    return failed;
}

bool Segmentation::isBadge(const cv::Mat& imageSegment)
{
    int* verticalHistogram = computeVerticalHistogram(imageSegment); //Bild von oben nach unten    

    verticalHistogram = reverseArray(verticalHistogram, 0,imageSegment.rows-1); //jetzt von unten nach oben
    plotArray(verticalHistogram, imageSegment.rows, "badgeProj.txt",false,true);

    int point1 = findPeak(verticalHistogram,imageSegment.rows-1,0,3);
    if(point1 >= 0) //evtl -1 od. -2 wenn nix gefunden wurde
        point1 = findValley(verticalHistogram,imageSegment.rows-1,point1,2);

    if(!isInInterval(point1,pair<int,int>(1,imageSegment.rows*0.5)))
        return false;

    int point2 = point1;
    while(verticalHistogram[point2] == verticalHistogram[point2+1])
        point2++;

    if((point2-point1 > imageSegment.rows*0.085) && (point1 < imageSegment.rows*0.6)) // Abstand hat gewisse Größe und ist beginnt auch in unterer Bildhälfte
        return true;
    else
        return false;
}

int Segmentation::findChange(int *horizontalHistogram, int start, int maxPos)
{
    //thresholdValley = threshold value, which should indicate beginning of a valley
    int result = -1;
    int i = start;


    while((horizontalHistogram[i+1] == horizontalHistogram[i]) && (i<maxPos)) //Abweichung finden
        i++;

    if(i > start+5)
        result = i;

    return result;
}

int Segmentation::findValley(int *horizontalHistogram, int size, int position, int thresholdValley)
{
    //thresholdValley = threshold value, which should indicate beginning of a valley
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

int Segmentation::findPeak(int *histogram, int size, int position, int thresholdPeak)
{
    // thresholdPeak = hreshold value, which should indicate beginning of a peak
    int result = -1;
    int i;

    plotArray(histogram, size, "findPeak.txt",false,false);

    for(i=position; i < size; i++){ //Punkt finden der Schwellwert überschreitet
        if(histogram[i] >= thresholdPeak){ //lok. Maximum finden
            while(histogram[i+1] >= histogram[i] && i<size) i++;
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5))
        result = -2; //Ende des Bildes erreicht?

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

    //if(horizontal)
        //imshow("Detect horizontal lines", cdst);
    //else
        //imshow("Detect vertical lines", cdst);


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
    //imshow("Rotiert", rotated);
    cout << "Nach dem Rotieren" << endl;
    //imshow("Nach dem Rotieren", computeBinaryImage(rotated, WOLFJOLION, WINDOW_SIZE));


    // Crop horizontal
    Mat horizontalCropped = cropHorizontal(rotated);
    cout << "Nach dem horizontal cropping" << endl;
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Vertical.txt' with linespoint\"");
    //system("gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/Horizontal.txt' with linespoint\"");


    if(horizontalCropped.rows != 0 && horizontalCropped.cols != 0){
        //imshow("Rotated horizontal cropped", horizontalCropped);

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
            v.first = 60/100.0 * 255;
            v.second = 255;

            if(isInInterval(hue,h) && isInInterval(saturation,s) && isInInterval(value,v)){
                //make it black
                pixel[2] = 0;
            }
        }
        cvtColor(hsvImage, horizontalCropped, CV_HSV2BGR);
        //imshow("blue to black", horizontalCropped);
        cout << "Nach dem Schwarzmachen" << endl;


        double angle = computeAngle(horizontalCropped, false);
        double slope = tan(angle*PI/180);
        Mat sheared = shear(horizontalCropped, slope);
        //imshow("Sheared", sheared);
        cout << "Nach dem Scheren" << endl;


        // Crop at the left and right side
        int start = getHorizontalStart(sheared);
        int end = getHorizontalEnd(sheared);
        if(start < end){
            croppedImage = sheared(Rect(start, 0, end-start, horizontalCropped.rows));
            //imshow("Cropped binary image", croppedBinaryImage);
            //imshow("Cropped Image", croppedImage);
            cout << "Nach dem gesamten Cropping" << endl;
        } else {
            croppedImage = sheared;
            //imshow("Cropped binary image", croppedBinaryImage);
            cout << "Vertical cropping failed" << endl;
        }
        cvtColor(computeBinaryImage(croppedImage, WOLFJOLION, 60), croppedBinaryImage, CV_GRAY2BGR);
        imwrite("Cropped/" + name, croppedImage);
        imwrite("Cropped_Binary/" + name, croppedBinaryImage);

        return croppedImage;

    } else {
        cerr << "Horizontal cropping failed" << endl;
    }

    cout << "No cropping happend - return originalImage" << endl;
    return image;
}


bool Segmentation::isInInterval(int value, std::pair<int,int> interval){
    return ((value >= interval.first) && (value <= interval.second));
}

void Segmentation::plotArray(int* array, int length, string filename, bool rm, bool view){
    std::ofstream myfile;
    myfile.open(filename);
    std::stringstream ss;

    for(int i = 0; i < length; i++){
        //ss << "(";
        //ss << i;
        //ss << ",";
        ss << array[i];
        //ss << ")";
        ss << ",\n";
    }
    myfile << ss.str();
    myfile.close();

    char* shellCmd = new char[256];

    shellCmd[0]=0;
    strcat(shellCmd,"gnuplot -p -e \"plot '/home/marius/Sciebo/Studium/9_WS15-16/3_CV-Praktikum/build-LPR-Desktop-Debug/");
    //gnuPlotCmd = gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/";
    const char* tmp = filename.c_str();
    strcat(shellCmd,tmp);
    strcat(shellCmd,"' with linespoint\"");
    strcat(shellCmd,"\0");

    if(view) system(shellCmd);

    if(rm){
        shellCmd[0]=0;
        strcat(shellCmd,"rm -f /home/marius/Sciebo/Studium/9_WS15-16/3_CV-Praktikum/build-LPR-Desktop-Debug/");
        strcat(shellCmd,tmp);
        system(shellCmd);
        strcat(shellCmd,"\0");
    }
}

int Segmentation::getHorizontalStart(const Mat& image){
    int* horizontalHistogram = computeHorizontalHistogram(image, WOLFJOLION);
    int width = image.cols;
    int middleRow = width/2;
    plotArray(horizontalHistogram, image.cols, "Horizontal.txt",false,false);

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
    plotArray(horizontalHistogram, image.cols, "Horizontal.txt",false,false);

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
    return indexAtMax;
}

int Segmentation::getVerticalStart(const Mat& image){
    int* verticalHistogram = computeVerticalHistogram(image);
    //very important: don't mix cols with rows -> bad results
    plotArray(verticalHistogram, image.rows, "Vertical.txt",false,false);

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
    plotArray(verticalHistogram, image.rows, "Vertical.txt",false,false);

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

/* Function to reverse an array*/
int* Segmentation::reverseArray(int *arr, int start, int end)
{
    int temp;
    while (start < end)
    {
        temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
    return arr;
}
