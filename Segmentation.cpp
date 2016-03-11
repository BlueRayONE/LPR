/*
 *
 *
 * This class contains the preprocessing for the segmentation and the segmenation itself
 *
 *
 */

#include "Segmentation.h"
#include "ImageViewer.h"
#include "binarizewolfjolion.h"

#include <iostream>
#include <fstream>
#include <string>
//#include <tesseract/ocrclass.h>

#define PI 3.14159265
#define WINDOW_SIZE 45

// enable DEV is for debugging usage
//#define DEV;

using namespace cv;
using namespace std;

Segmentation::Segmentation(const Mat& image): originalImage(image){
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

    int* colProjection = computeColProjection(croppedImage, WOLFJOLION);

    #ifdef DEV
        plotArray(colProjection, size, "horizontalFULL.txt",false,true);
    #endif

    for(int charNo=0; charNo<11; charNo++){
        if(rightPos >= size-10) break;

        leftPos = rightPos; // End of prev elem is start of new elem
        tmpPos=findPeak(colProjection,size,leftPos,30); //Peak des nächsten Elements finden

        if(tmpPos == -1){ //Keinen Peak gefunden!!
            failed = true;
            break;
        }
        else if(tmpPos == -2) break;//Ende erreicht


        rightPos=findValley(colProjection,size,tmpPos,3); //Ende des nächsten Elements finden

        if(rightPos == -1){ //Kein Valley gefunden!!
            failed = true;
            break;
        }
        else if(rightPos == -2) break; //Ende erreicht

        rightPos += 2; // Leicht erhöhen um wirklich im weißen Bereich NACH Buchstaben zu sein.

        //Badge muss innerhalb der ersten 4 Iterationen gefunden worden sein => charNo <= 3
        if(badgeFound || charNo > 3 || !isBadge(croppedImage(Rect(leftPos, 0, rightPos-leftPos, croppedImage.rows)))){ //Es handelt sich nicht um Bereich der Plaketten
            line(croppedImage, cv::Point(rightPos, 0), Point(rightPos, croppedImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende des Buchstabens einzeichnen
            tmpPos = findChange(colProjection,leftPos+2,rightPos);
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
            rightPos=findValley(colProjection,size,tmpPos,3) + 2;
            line(croppedImage, cv::Point(rightPos, 0), Point(rightPos, croppedImage.rows), Scalar(255, 0, 0), 1, CV_AA); // Ende der Plakette einzeichnen
        }


    }

    ImageViewer::viewImage(croppedImage, "segmented", 400);
    return failed;
}

bool Segmentation::isBadge(const cv::Mat& imageSegment)
{
    if(isBadgeDetail(imageSegment,true)){
        return true;
    }else{
        return isBadgeDetail(imageSegment,false);
    }
}

bool Segmentation::isBadgeDetail(const cv::Mat& imageSegment, bool reverse)
{
    int* rowProjection = computeRowProjection(imageSegment); //Bild von oben nach unten

    if(reverse) rowProjection = reverseArray(rowProjection, 0,imageSegment.rows-1); //jetzt von unten nach oben
    #ifdef DEV
        plotArray(rowProjection, imageSegment.rows, "badgeProj.txt",false,true);
    #endif

    int point1 = findPeak(rowProjection,imageSegment.rows-1,0,3);
    if(point1 >= 0) //evtl -1 od. -2 wenn nix gefunden wurde
        point1 = findValley(rowProjection,imageSegment.rows-1,point1,2);

    if(!isInInterval(point1,pair<int,int>(1,imageSegment.rows*0.5)))
        return false;

    int point2 = point1;
    while(rowProjection[point2] == rowProjection[point2+1])
        point2++;

    if((point2-point1 > imageSegment.rows*0.085) && (point1 < imageSegment.rows*0.6)) // Abstand hat gewisse Größe und ist beginnt auch in unterer Bildhälfte
        return true;
    else
        return false;
}

int Segmentation::findChange(int* colProjection, int start, int maxPos)
{
    //thresholdValley = threshold value, which should indicate beginning of a valley
    int result = -1;
    int i = start;


    while((colProjection[i+1] == colProjection[i]) && (i<maxPos)) //Abweichung finden
        i++;

    if(i > start+5)
        result = i;

    return result;
}

int Segmentation::findValley(int* colProjection, int size, int position, int thresholdValley)
{
    //thresholdValley = threshold value, which should indicate beginning of a valley
    int result = -1;
    int i;

    for(i=position; i < size; i++){ //Punkt finden der Schwellwert unterschreitet
        if(colProjection[i] <= thresholdValley){
            while(colProjection[i+1] < colProjection[i] && i<size) i++; //lok. Minimum finden
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5)) result = -2; //Ende des Bildes erreicht?

    return result;
}

int Segmentation::findPeak(int *projection, int size, int position, int thresholdPeak)
{
    // thresholdPeak = hreshold value, which should indicate beginning of a peak
    int result = -1;
    int i;

    #ifdef DEV
        plotArray(projection, size, "findPeak.txt",false,false);
    #endif

    for(i=position; i < size; i++){ //Punkt finden der Schwellwert überschreitet
        if(projection[i] >= thresholdPeak){ //lok. Maximum finden
            while(projection[i+1] >= projection[i] && i<size) i++;
            result = i;
            break;
        }
    }

    if((result == -1) && (i>size-5))
        result = -2; //Ende des Bildes erreicht?

    return result;
}

/**
 * @brief Computes the angle of the licence plate.
 * @param image : a colored picture where a licence plate can be seen.
 * @return the angle of the licence plate
 */
double Segmentation::computeAngle(const Mat& image){
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION, WINDOW_SIZE);

    Mat dst, cdst;
    // Do Canny for edge detection
    Canny(binaryImage, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;

    // Do Hough for line detection
    int threshold = 100;
    int minLinLength = 1/2 * cdst.cols;
    HoughLinesP(dst, lines, 1, CV_PI/180, threshold, minLinLength, 20);

    int lineIndex = 0;
    double angle = 0; // default angle for a licence plate
    double maxLength = 0;
    bool found = false;

    // find the longest line and remember its angle
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

    // if a line was real line was found paint it in the picture
    if(found){
        Vec4i l = lines[lineIndex];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
    }

    imwrite("RotatedWithLine.png", cdst);
    return angle;
}

/**
 * @brief Computes the shearing angle of the characters.
 * @param image : a colored image where a line of characters can be seen.
 * @return the shearing angle
 */
double Segmentation::computeShearingAngle(const Mat& blackened){
    Mat binaryImage = computeBinaryImage(blackened, WOLFJOLION, WINDOW_SIZE);

    Mat dst, cdst;
    // Do Canny for edge detection
    Canny(binaryImage, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;

    // Do Hough for line detection
    int threshold = cdst.rows * 0.5;
    int minLinLength = 1/3 * cdst.rows;
    HoughLinesP(dst, lines, 1, CV_PI/180, threshold, minLinLength, 20);

    int lineIndex = 0;
    double shearingAngle = 90; // default angle for character angle
    double maxLength = 0;
    bool found = false;

    // find the longest line and remember its angle
    for(size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);

        double currentSlope = (double)(l[3] - l[1]) / (l[2] - l[0]);
        double currentAngle = atan(currentSlope) * 180/PI;
        double currentLength = sqrt(pow((l[0] - l[2]), 2) + pow((l[1] - l[3]), 2));

        if((currentLength > maxLength) && (abs(currentAngle) > 45)){
            maxLength = currentLength;
            lineIndex = i;
            shearingAngle = currentAngle;
            found = true;
        }
    }

    // if a line was real line was found paint it in the picture
    if(found){
        Vec4i l = lines[lineIndex];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
    }

    imwrite("ShearedWithLine.png", cdst);
    return shearingAngle;
}

/**
 * @brief Rotates the image. Used to rotate the licence plate.
 * @param toRotate : the image that will be rotated.
 * @return a rotated image
 */
Mat Segmentation::rotate(const cv::Mat& toRotate){
    double angle = computeAngle(toRotate);

    Point2f pt(toRotate.cols/2., toRotate.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    Mat rotated = Mat(toRotate.rows, toRotate.cols, toRotate.type());

    // here, the rotation happens
    warpAffine(toRotate, rotated, r, Size(toRotate.cols, toRotate.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
    return rotated;
}


/**
 * @brief Binarizes the given image with a given binarization method.
 * @param image : the image
 * @param version : binarization method
 * @param windowSize : window size is dependent of picture size and must be determined for different sized images.
 * @return a binary version of the image
 */
Mat Segmentation::computeBinaryImage(Mat image, NiblackVersion version, int windowSize){
    Mat greyImage;
    cvtColor(image, greyImage, CV_BGR2GRAY);
    Mat binaryImage(greyImage.rows, greyImage.cols, CV_8UC1);

    // here the code of Wolf et al. is used
    NiblackSauvolaWolfJolion(greyImage, binaryImage, version, windowSize, windowSize, 0.5, 128);
    return binaryImage;
}

/**
 * @brief Crops the given image from the top and from the bottom to best fitting.
 * @param image : the image that will be cropped.
 * @return an image that is cropped from the top and bottom.
 */
Mat Segmentation::cropHorizontal(const Mat& image){
    // Search the rows which enclose the the characters
    int start = getVerticalStart(image);
    int end = getVerticalEnd(image);

    // sanity check
    if(start < end){
        Mat horizontalCropped = image(Rect(0, start, image.cols, end-start));
        return horizontalCropped;
    } else
        return image;
}


/**
 * @brief Shears the given image to eliminate shearing angle of text in the image.
 * @param blackened : an image with a licence plate where the euroline is blackened
 * @return a sheared image
 */
Mat Segmentation::shear(const Mat& blackened){
    double angle = computeShearingAngle(blackened);
    double slope = tan(angle*PI/180);

    Mat warpedImage = Mat(blackened.rows, blackened.cols, blackened.type());
    Mat shearMat = (Mat_<double>(3,3) << 1, -1/slope, 0, 0, 1, 0, 0, 0, 1);
    Size size(blackened.cols, blackened.rows);
    warpPerspective(blackened, warpedImage, shearMat, size);

    return warpedImage;
}

/**
 * @brief Blackens the euroline in the image. Used to determine a left border.
 * @param horizontalCropped
 */
void Segmentation::blackenEuroline(Mat& horizontalCropped){
    // make all blue parts black
    Mat hsvImage;
    cvtColor(horizontalCropped, hsvImage, CV_BGR2HSV);
    MatIterator_<Vec3b> it = hsvImage.begin<Vec3b>();
    MatIterator_<Vec3b> it_end = hsvImage.end<Vec3b>();
    for(; it != it_end; ++it){
        Vec3b& pixel = *it; // reference to pixel in image
        int hue = pixel[0];
        int saturation = pixel[1];
        int value = pixel[2];

        pair<int,int> h, s, v;
        h.first = 90;
        h.second = 130;
        s.first = 50/100.0 * 255;
        s.second = 255;
        v.first = 60/100.0 * 255;
        v.second = 255;

        if(isInInterval(hue,h) && isInInterval(saturation,s) && isInInterval(value,v)){
            //make it black
            pixel[2] = 0; // value = 0
        }
    }
    cvtColor(hsvImage, horizontalCropped, CV_HSV2BGR);
}

/**
 * @brief Central cropping method which coordinates the single preprocessing methods. Crops the image with a licence plate on it to an ideal image ready for segmentation.
 * @param image : an image in which the licence plate is nearly in the middle
 * @return a fully cropped image (rotated, sheared, vertical and horizontal cropped)
 */
Mat Segmentation::cropImage(const Mat& image){
    // Filter the image to sharpen the edges and remove noise
    Mat filteredImage = Mat(image.rows, image.cols, image.type());
    bilateralFilter(image, filteredImage, 9, 100, 1000);

#ifdef DEV
    imwrite("Filtered.png", filteredImage);
    cout << "After filtering" << endl;
#endif

    // Rotate the image
    Mat rotated = rotate(filteredImage);
#ifdef DEV
    imwrite("Rotated.png", rotated);
    cout << "After rotating" << endl;
    imwrite("RotatedBinary.png", computeBinaryImage(rotated, WOLFJOLION, WINDOW_SIZE));
#endif

    // Crop horizontal
    Mat horizontalCropped = cropHorizontal(rotated);
#ifdef DEV
    imwrite("HorizontalCropped.png", horizontalCropped);
    cout << "After horizontal cropping" << endl;
#endif

    if(horizontalCropped.rows != 0 && horizontalCropped.cols != 0){
#ifdef DEV
        imshow("Rotated horizontal cropped", horizontalCropped);
#endif

        blackenEuroline(horizontalCropped);
#ifdef DEV
        imwrite("Blackened.png", horizontalCropped);
        cout << "After blackened euroline" << endl;
#endif

        Mat sheared = shear(horizontalCropped);
#ifdef DEV
        imwrite("Sheared.png", sheared);
        cout << "After shearing" << endl;
#endif
        // Crop at the left and right side
        int start = getHorizontalStart(sheared);
        int end = getHorizontalEnd(sheared);

        // sanity check
        if(start < end){
            croppedImage = sheared(Rect(start, 0, end-start, horizontalCropped.rows));
#ifdef DEV
            cout << "After full cropping" << endl;
#endif
        } else {
            croppedImage = sheared;
#ifdef DEV
            cerr << "Vertical cropping failed" << endl;
#endif
        }
        // finally binarize the fully cropped image for future use
        cvtColor(computeBinaryImage(croppedImage, WOLFJOLION, 60), croppedBinaryImage, CV_GRAY2BGR);
#ifdef DEV
        imwrite("Cropped.png", croppedImage);
        imwrite("CroppedBinary.png", croppedBinaryImage);
#endif
        return croppedImage;

    } else {
        cerr << "Horizontal cropping failed" << endl;
    }

    cout << "No cropping happend - return originalImage" << endl;
    return image;
}


/**
 * @brief Utility method that checks if a number is in an interval
 * @param value : the value
 * @param interval : the interval
 * @return
 */
bool Segmentation::isInInterval(int value, std::pair<int,int> interval){
    return ((value >= interval.first) && (value <= interval.second));
}

/* Utility function that plots a projection */
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
    // do a gnuplot command - define a path to your Qt debug folder !!!!
    //strcat(shellCmd,"gnuplot -p -e \"plot '/home/marius/Sciebo/Studium/9_WS15-16/3_CV-Praktikum/build-LPR-Desktop-Debug/");
    //strcat(shellCmd,"gnuplot -p -e \"plot '/home/alex/Documents/build-LPR-Desktop_Qt_5_5_1_GCC_64bit-Debug/");
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

/**
 * @brief Computes the left border.
 * @param image
 * @return the index of the col which represents the left border.
 */
int Segmentation::getHorizontalStart(const Mat& image){
    int* colProjection = computeColProjection(image, WOLFJOLION);
    int width = image.cols;
    int middleRow = width/2;
    #ifdef DEV
        plotArray(colProjection, image.cols, "ColProjection.txt",false,false);
    #endif

    int maxValue = 0;
    int indexAtMax = 0;
    // start from the middle col and search till the first col
    // find index with maximum value
    for(int i = middleRow; i >= 0; i--){
        int currentValue = colProjection[i];

        if(currentValue > maxValue){
            maxValue = currentValue;
            indexAtMax = i;
        }
    }
    delete colProjection;
    return indexAtMax + 5;
}

/**
 * @brief Computes the right border.
 * @param image
 * @return the index of the col which represents the right border
 */
int Segmentation::getHorizontalEnd(const Mat& image){
    int* colProjection = computeColProjection(image, WOLFJOLION);
    int width = image.cols;
    int middleRow = width/2;
    #ifdef DEV
        plotArray(colProjection, image.cols, "ColProjection.txt",false,false);
    #endif

    int indexAtMax = 0;
    //start from the middle col and search till the end col
    // find index with maximum value
    for(int i = middleRow; i < width; i++){
        int currentValue = colProjection[i];

        indexAtMax = i;
        if(currentValue == image.rows){
            return indexAtMax;
        }
    }
    delete colProjection;
    return indexAtMax;
}

/**
 * @brief Computes the top border.
 * @param image
 * @return the index of the row which represents the top border.
 */
int Segmentation::getVerticalStart(const Mat& image){
    int* rowProjection = computeRowProjection(image);
    //very important: don't mix cols with rows -> bad results
    #ifdef DEV
        plotArray(rowProjection, image.rows, "RowProjection.txt",false,true);
    #endif

    int height = image.rows;
    int middle = height/2;
    int startIndex = middle;

    // must be dynamic (7 seems to be a good factor)
    int threshold = image.cols / 7;

    // start from the middle row and search till the first row
    int maximum = rowProjection[startIndex];
    bool found = false;
    int i = 1;
    while(!found){
        int currentValue = rowProjection[startIndex - i];

        if(currentValue < maximum){
            int diff = maximum - currentValue;
            if(diff >= threshold){
                found = true;
                int index = startIndex - i;
                int currentMin = rowProjection[index];
                int neighborMin = rowProjection[index - 1];
                while(neighborMin < currentMin){
                    index = index - 1;
                    currentMin = rowProjection[index];
                    neighborMin = rowProjection[index - 1];
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

    delete rowProjection;
    if(isInInterval(startIndex, pair<int,int>(0, image.rows-1))){
        return startIndex;
    }
    else
        return 0;
}

/**
 * @brief Computes the bottom border.
 * @param image
 * @return the index of the row that represents the bottom border
 */
int Segmentation::getVerticalEnd(const Mat& image){
    int* rowProjection = computeRowProjection(image);
    //very important: don't mix cols with rows -> bad results
    #ifdef DEV
        plotArray(rowProjections, image.rows, "RowProjection.txt",false,false);
    #endif

    int height = image.rows;
    int middle = height/2;
    int endIndex = middle;

    // must be dynamic
    int threshold = image.cols / 7;

    // start from the middle row and search till the first row
    int maximum = rowProjection[endIndex];
    bool found = false;
    int i = 1;
    while(!found){
        int currentValue = rowProjection[endIndex + i];

        if(currentValue < maximum){
            int diff = maximum - currentValue;
            if(diff >= threshold){
                found = true;
                int index = endIndex + i;
                int currentMin = rowProjection[index];
                int neighborMin = rowProjection[index + 1];
                while(neighborMin < currentMin){
                    index = index + 1;
                    currentMin = rowProjection[index];
                    neighborMin = rowProjection[index + 1];
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

    delete rowProjection;
    if(isInInterval(endIndex, pair<int,int>(0, image.rows-1))){
        return endIndex;
    }
    else
        return 0;
}

/**
 * @brief Computes the col projection of the given image.
 * @param image
 * @param version : the binarization method which will be used.
 * @return an array that contains all the y-values for each col. Must be FREED!
 */
int* Segmentation::computeColProjection(const Mat& image, NiblackVersion version){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, version, WINDOW_SIZE);

    int* colProjection = new int[width];
    for(int i = 0; i < width; i++){
       colProjection[i] = height - countNonZero(binaryImage.col(i));
    }
    return colProjection;
}

/**
 * @brief Computes the row projection of the given image.
 * @param image
 * @return an array that contains all the y-values for each row. Must be FREED!
 */
int* Segmentation::computeRowProjection(const Mat& image){
    int width = image.cols;
    int height = image.rows;
    Mat binaryImage = computeBinaryImage(image, WOLFJOLION, WINDOW_SIZE);

    int* rowProjection = new int[height];
    for(int i = 0; i < height; i++){
       rowProjection[i] = width - countNonZero(binaryImage.row(i));
    }
    return rowProjection;
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
