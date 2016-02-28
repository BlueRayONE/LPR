#include "classification.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "binarizewolfjolion.h"

using namespace std;
using namespace cv;
using namespace cv::text;

Classification::Classification()
{

}


void Classification::characterRecognition(const Mat& image){
    Segmentation segmentation(image);
    segmentation.cropImage(image);
    vector<cv::Mat> chars = Segmentation_MSER::findChars(segmentation.croppedImage);

    /*Ptr<OCRHMMDecoder::ClassifierCallback> ocr = loadOCRHMMClassifierCNN("/home/alex/opencv_contrib/modules/text/samples/OCRHMM_knn_model_data.xml.gz");

    // for all the characters
    for(int i = 0; i < chars.size(); i++){

        //cout << "A demo program of Scene Text Character Recognition: " << endl;
        //cout << "Shows the use of the OCRBeamSearchDecoder::ClassifierCallback class using the Single Layer CNN character classifier described in:" << endl;
        //cout << "Coates, Adam, et al. \"Text detection and character recognition in scene images with unsupervised feature learning.\" ICDAR 2011." << endl << endl;

        // the image must contain a single character
        Mat character = chars.at(i);
        string vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the clasifier output classes

        double t_r = (double)getTickCount();
        vector<int> out_classes;
        vector<double> out_confidences;

        ocr->eval(image, out_classes, out_confidences);

        cout << "OCR output = \"" << vocabulary[out_classes[0]] << "\" with confidence "
                                                                << out_confidences[0] << ". Evaluated in "
                                                                << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl << endl;
    }*/

    Ptr<OCRTesseract> tesseract = OCRTesseract::create(NULL, "deu", NULL, NULL, 10);

    string result = "";
    for(int i = 0; i < chars.size(); i++){
        Mat character = chars.at(i);

        string output;
        vector<Rect> boxes;
        vector<string> words;
        vector<float> confidences;
        tesseract->run(character, output, &boxes, &words, &confidences);
        output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());
        result.append(output);
    }
    cout << result << endl;
}


