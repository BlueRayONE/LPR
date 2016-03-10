#include "Classification.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "binarizewolfjolion.h"

using namespace std;
using namespace cv::text;

Classification::Classification(){
}

/**
 * @brief Recognizes the characters on the given plates.
 * @param plates : images where a licence plate is on every picture
 * @param projection : true for projection segmentation, false for MSER segmentation
 * @return a vector of strings of the licence plates
 */
vector<string> Classification::characterRecognition(const vector<cv::Mat> plates, bool projection){
    //define your path to tessdata folder where "europlate.traineddata" can be found
    // last parameter "10" defines that the given matrix represent a single character (psmode 10)
    cv::Ptr<OCRTesseract> tesseract = OCRTesseract::create("/usr/local/share/tessdata", "europlate", NULL, 0, 10);
    vector<string> results;

    // for every found plate do a recognition
    for(int k = 0; k < plates.size(); k++){
        cv::Mat plate = plates.at(k);
        vector<cv::Mat> chars;

        //crop the plate as preprocessing for the segmentation
        Segmentation segmentation(plate);
        segmentation.cropImage(plate);

        if(!projection){
            Segmentation_MSER seg_mser = Segmentation_MSER(segmentation.croppedImage);
            //the single character matrices
            chars = seg_mser.findChars();
        } else {
            segmentation.findChars();
            // the single character matrices
            chars = segmentation.chars;
        }

        bool numbers = false;
        string result = "";
        for(int i = 0; i < chars.size(); i++){
            cv::Mat character = chars.at(i);

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

            // do some postprocessing where letters between numbers will be excluded from result
            if(numbers){
                if(asciicode < 58)
                    result.append(output);
            } else {
                result.append(output);
            }
        }
        results.push_back(result);
    }

    // every string in "results" represents a single licence plate
    return results;
}


