#include "classification.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "binarizewolfjolion.h"

using namespace std;
using namespace cv::text;

Classification::Classification(){
}


vector<string> Classification::characterRecognition(const vector<cv::Mat> plates, bool projection){
    cv::Ptr<OCRTesseract> tesseract = OCRTesseract::create("/usr/local/share/tessdata", "deu2", NULL, 0, 10);
    vector<string> results;

    for(int k = 0; k < plates.size(); k++){
        cv::Mat plate = plates.at(k);
        vector<cv::Mat> chars;

        Segmentation segmentation(plate);
        segmentation.cropImage(plate);

        if(!projection){
            Segmentation_MSER seg_mser = Segmentation_MSER(segmentation.croppedImage);
            chars = seg_mser.findChars();
        } else {
            segmentation.findChars();
            chars = segmentation.chars;
        }

        bool numbers = false;
        bool finished = false;
        string result = "";
        for(int i = 0; i < chars.size(); i++){
            cv::Mat character = chars.at(i);

            //Mat binchar = segmentation.computeBinaryImage(character, WOLFJOLION, 30);
            //cv::imshow(to_string(i), character);
            string output;
            vector<cv::Rect> boxes;
            vector<string> words;
            vector<float> confidences;
            tesseract->run(character, output, &boxes, &words, &confidences);
            output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());

            int asciicode = (output.c_str())[0];
            if(asciicode < 58 && asciicode != 32){
                numbers = true;
            }

            if(numbers){
                if(asciicode < 58)
                    result.append(output);
            } else {
                result.append(output);
            }

            for(int j = 0; j < confidences.size(); j++){
                cout << confidences.size() << endl;
                cout <<  "Char " + to_string(i) << " " << confidences.at(j) << " " << words.at(j) << endl;
            }
        }
        results.push_back(result);
    }

    return results;
}


