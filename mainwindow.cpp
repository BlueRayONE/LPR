#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "localisation.h"
#include "segmentation.h"
#include "licencePlateRecognition.hpp"
#include <iostream>
#include <fstream>
#include <string>

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

    ui->radio_mser->setChecked(true);
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
		cv::Mat imgX = originalImage.clone();
		//cv::Mat imgY = originalImage.clone();

		cv::Mat Plate = a.pca(imgX);
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
	Segmentation::segmentationTest(originalImage);
}

void MainWindow::on_btn_segment_clicked()
{
    if(ui->radio_projection->isChecked())
    {
            Segmentation segmentation(originalImage);

            cv::Mat croppedImage = segmentation.cropImage(originalImage);
            segmentation.findChars(croppedImage);
    }
    else
    {
        //Segmentation segmentation(originalImage);
        //cv::Mat cropped = segmentation.cropImage(originalImage);
        Segmentation_MSER::findChars(originalImage);
    }
}
