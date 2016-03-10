#-------------------------------------------------
#
# Project created by QtCreator 2015-12-04T16:04:34
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = LPR
TEMPLATE = app


SOURCES += main.cpp\
    ImageReader.cpp \
    QtOpencvCore.cpp \
    ImageViewer.cpp \
    Wavelet.cpp \
    binarizewolfjolion.cpp \
    MSER.cpp \
    Segmentation_MSER.cpp \
    PCA_Localisation.cpp \
    Segmentation.cpp \
    MainWindow.cpp \
    Classification.cpp

HEADERS  += \
    ImageReader.hpp \
    QtOpencvCore.hpp \
    ImageViewer.h \
    daub.h \
    Wavelet.h \
    binarizewolfjolion.h \
    MSER.h \
    Segmentation_MSER.h \
    PCA_Localisation.h \
    Segmentation.h \
    MainWindow.h \
    Classification.h

FORMS    += mainwindow.ui


win32 {
    INCLUDEPATH += $$(OPENCV_DIR)/include

    Debug:LIBS +=  -L$$(OPENCV_DIR)/x86/vc12/lib \
                    -lopencv_core310d \
                    -lopencv_highgui310d \
                    -lopencv_imgproc310d \
                    -lopencv_imgcodecs310d \
                    -lopencv_text310d \
                    -lopencv_features2d310d
        LIBS += -L"D:\tesseract\tesseract-ocr\vs2013\bin\Win32\DLL_Debug" \
                -llibtesseract304d
    Release:LIBS +=  -L$$(OPENCV_DIR)/x86/vc12/lib \
                    -lopencv_core310 \
                    -lopencv_highgui310 \
                    -lopencv_imgproc310 \
                    -lopencv_imgcodecs310 \
                    -lopencv_text310 \
                    -lopencv_features2d310

}

unix {

    QMAKE_CXXFLAGS += -std=c++11 -Wall -pedantic -Wno-unknown-pragmas

    INCLUDEPATH += /opt/opencv3/include/ \
                   /usr/local/include

    LIBS += -L/opt/opencv3/lib/ \
            -L/usr/local/lib \
            -L/usr/lib \
            -ltesseract \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc \
            -lopencv_imgcodecs \ 
            -lopencv_text \
            -lopencv_features2d
    QMAKE_CXXFLAGS_WARN_ON = -Wno-unused-variable -Wno-reorder
}
