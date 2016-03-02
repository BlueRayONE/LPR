#include "classification.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "binarizewolfjolion.h"

using namespace std;
using namespace cv::text;

Classification::Classification(const cv::Mat& image, string filename) : originalImage(image), filename(filename)
{

}


void Classification::characterRecognition(const cv::Mat& image){
    MSER mser(image);
    vector<cv::Mat> plates = mser.run();
    cv::Mat plate = plates.at(0);

    Segmentation segmentation(plate, filename);
    segmentation.cropImage(plate);
    Segmentation_MSER seg_mser = Segmentation_MSER(segmentation.croppedImage);
    vector<cv::Mat> chars = seg_mser.findChars();

    //segmentation.findChars();
    //vector<cv::Mat> chars = segmentation.chars;

    cv::Ptr<OCRTesseract> tesseract = OCRTesseract::create("/usr/local/share/tessdata", "deu2", NULL, 0, 10);

    string result = "";
    for(int i = 0; i < chars.size(); i++){
        cv::Mat character = chars.at(i);

        //Mat binchar = segmentation.computeBinaryImage(character, WOLFJOLION, 30);

        cv::imshow(to_string(i), character);
        string output;
        vector<cv::Rect> boxes;
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


