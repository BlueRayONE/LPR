#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>

#include "ImageReader.hpp"
#include "ImageViewer.h"
#include "QtOpencvCore.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Wavelet.h"
#include "MSER.h"
#include "Segmentation_MSER.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_btn_openImage_clicked();
    void on_btn_localize_clicked();

    //cv::Mat lprThreshold(cv::Mat inputImg);

    void on_btn_crop_clicked();

    void on_btn_segment_clicked();

    void on_btn_recognize_clicked();

private:
    Ui::MainWindow *ui;
    void enableGUI();
    void disableGUI();
    cv::Mat originalImage;
	std::vector<cv::Mat> locatedCandidates;
    std::string name;
};

#endif // MAINWINDOW_H
