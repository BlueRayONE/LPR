#include "mainwindow.h"
#include "ui_mainwindow.h"

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
    QString imagePath = QFileDialog::getOpenFileName(this, "Open Image...", QString(), QString("Images *.png *.jpg *.tiff *.tif"));

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
           ImageViewer::viewImage(originalImage, "Original Image", 800);
       }
    }

}


void MainWindow::enableGUI()
{
    ui->btn_localize->setEnabled(true);


}

void MainWindow::disableGUI()
{
    ui->btn_localize->setEnabled(false);
}

void MainWindow::on_btn_localize_clicked()
{
    //localize
}
