#include "classification.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "binarizewolfjolion.h"

using namespace std;
using namespace cv;
using namespace cv::text;

Classification::Classification(const Mat& image, string filename) : originalImage(image), filename(filename)
{

}


void Classification::characterRecognition(const Mat& image){
    Segmentation segmentation(image, filename);
    segmentation.cropImage(image);
    vector<cv::Mat> chars = Segmentation_MSER::findChars(segmentation.croppedImage);
    //segmentation.findChars();
    //vector<cv::Mat> chars = segmentation.chars;

    Ptr<OCRTesseract> tesseract = OCRTesseract::create("/usr/local/share/tessdata", "deu2", NULL, 0, 10);

    string result = "";
    for(int i = 0; i < chars.size(); i++){
        Mat character = chars.at(i);

        //Mat binchar = segmentation.computeBinaryImage(character, WOLFJOLION, 30);

        //imshow(to_string(i), binchar);
        string output;
        vector<Rect> boxes;
        vector<string> words;
        vector<float> confidences;
        tesseract->run(character, output, &boxes, &words, &confidences);
        output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());
        result.append(output);
        for(int j = 0; j < confidences.size(); j++){
            cout << confidences.size() << endl;
            cout <<  "Char " + to_string(i) << " " << confidences.at(j) << " " << words.at(j) << endl;
        }

    }
    cout << result << endl;


    /*Ptr<OCRTesseract> tesseract = OCRTesseract::create("/usr/local/share/tessdata", "leu", NULL, NULL, 7);
    string output;
    vector<Rect> boxes;
    vector<string> words;
    vector<float> confidences;
    tesseract->run(segmentation.croppedBinaryImage, output, &boxes, &words, &confidences);
    output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());
    cout << output << endl;*/
}


