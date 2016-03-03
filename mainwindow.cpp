#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "localisation.h"
#include "segmentation.h"
#include "licencePlateRecognition.hpp"
#include "classification.h"
#include <iostream>
#include <fstream>
#include <string>

//#define DEV;

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    disableGUI();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_openImage_clicked()
{
	 //QString imagePath = "";
    QString imagePath = QFileDialog::getOpenFileName(this, "Open Image...", QString(), QString("Images *.png *.jpg *.tiff *.tif *.JPG"));

    if(!imagePath.isNull() && !imagePath.isEmpty())
    {
       cv::Mat img = ImageReader::readImage(QtOpencvCore::qstr2str(imagePath));

       // get file name
       string filename = imagePath.toStdString();
       const size_t last_slash_idx = filename.find_last_of('/');

       if (string::npos != last_slash_idx) {
           filename.erase(0, last_slash_idx + 1);
           name = filename;
       }

       if(!img.empty())
       {
           originalImage = img;
           enableGUI();

           // ImageViewer::viewImage(cv::Mat img, std::string title, int height = -1);
           //-1 --> original Size

           //shows img in full size
           //ImageViewer::viewImage(img, "Original Image");

           //resizes img to height 800 maintaining aspect ratio

           ImageViewer::viewImage(originalImage, "Original Image", 300);
        }
    }


}


void MainWindow::enableGUI()
{
    ui->btn_localize->setEnabled(true);
    ui->group_localization->setEnabled(true);
    ui->group_segmentation->setEnabled(true);
	ui->btn_crop->setEnabled(true);
	ui->btn_segment->setEnabled(true);

    ui->radio_pca->setChecked(true);
    ui->radio_projection->setChecked(true);


}

void MainWindow::disableGUI()
{
    ui->btn_localize->setEnabled(false);
    ui->group_localization->setEnabled(false);
    ui->group_segmentation->setEnabled(false);
	ui->btn_crop->setEnabled(false);
	ui->btn_segment->setEnabled(false);
}

void MainWindow::on_btn_localize_clicked()
{
    if(ui->radio_pca->isChecked())
    {

		licencePlateRecognition a = licencePlateRecognition();
		cv::Mat cloned = originalImage.clone();
        a.getPlate(cloned);
    }
    else if(ui->radio_mser->isChecked())
    {
		MSER m =  MSER(originalImage);
		locatedCandidates = m.run();
    }
    else
    {
        Wavelet* h = new Wavelet();
        h->run(originalImage);
    }
}

/*cv::Mat MainWindow::lprThreshold(cv::Mat inputImg)
{
    cv::Mat biFiImg, greyIm, eightBIm, thresholdIm;

    cv::bilateralFilter(inputImg, biFiImg, 18, 100, 1000, cv::BORDER_DEFAULT);

    cv::cvtColor(biFiImg, greyIm, CV_BGR2GRAY);

    greyIm.convertTo(eightBIm, CV_8UC1);

    //cv::adaptiveThreshold(eightBIm,thresholdIm,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,15,5);

    //cv::threshold(eightBIm, thresholdIm, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    cv::threshold(eightBIm, thresholdIm, 170, 255, CV_THRESH_BINARY);

    return thresholdIm;
}*/





void MainWindow::on_btn_crop_clicked()
{
    Segmentation segmentation(originalImage);
    segmentation.cropImage(originalImage);
}

void MainWindow::on_btn_segment_clicked()
{
    Segmentation segmentation(originalImage);
    cv::Mat cropped = segmentation.cropImage(originalImage);
    if(ui->radio_projection->isChecked())
    {
        segmentation.findChars();

        for(int i=0; i < segmentation.chars.size(); i++){
            char* title = new char[128];
            sprintf(title,"Buchstabe%i",i);
            #ifdef DEV
                 ImageViewer::viewImage(segmentation.chars[i], title, 400);
            #endif
            strcat(title,".jpg");
            #ifdef DEV
                imwrite(title, segmentation.chars[i]);
            #endif
        }
    }
    else
    {
		Segmentation_MSER segmentationMSER(cropped);
		segmentationMSER.findChars();
    }
}



void MainWindow::on_btn_recognize_clicked()
{
    vector<cv::Mat> plates;
    if(ui->radio_mser->isChecked()){
        MSER mser(originalImage);
        plates = mser.run();
    }

    if(ui->radio_pca->isChecked()){
        licencePlateRecognition a = licencePlateRecognition();
        cv::Mat cloned = originalImage.clone();
        cv::Mat plate = a.getPlate(cloned);
        plates.push_back(plate);
    }

    Classification classification;
    vector<string> recognized;
    if(ui->radio_mser_seg->isChecked()){
        recognized = classification.characterRecognition(plates, false);
    }

    if(ui->radio_projection->isChecked()){
        recognized = classification.characterRecognition(plates, true);
    }

    string output = "";
    for(string sample : recognized){
        output.append(sample + "\n");
    }

    QMessageBox::information(
        this, "Gefundene Kennzeichen",
        tr(output.c_str()));
}
