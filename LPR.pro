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
        mainwindow.cpp \
    ImageReader.cpp \
    QtOpencvCore.cpp \
    ImageViewer.cpp \
    Wavelet.cpp \
    segmentation.cpp \
    binarizewolfjolion.cpp \
    MSER.cpp

HEADERS  += mainwindow.h \
    ImageReader.hpp \
    QtOpencvCore.hpp \
    ImageViewer.h \
    daub.h \
    Wavelet.h \
    segmentation.h \
    binarizewolfjolion.h \
    MSER.h

FORMS    += mainwindow.ui


win32 {
    #QMAKE_CXXFLAGS += -std=c++11 -Wall -pedantic -Wno-unknown-pragmas

    INCLUDEPATH += $$(OPENCV_DIR)/include

    Debug:LIBS +=  -L$$(OPENCV_DIR)/x86/vc12/lib \
                -lopencv_core2411d \
                -lopencv_highgui2411d \
                -lopencv_imgproc2411d \
                -lopencv_features2d2411d
    Release:LIBS +=  -L$$(OPENCV_DIR)/x86/vc12/lib \
            -lopencv_core2411d \
            -lopencv_highgui2411d \
            -lopencv_imgproc2411d \
            -lopencv_features2d2411d


}

unix {

    QMAKE_CXXFLAGS += -std=c++11 -Wall -pedantic -Wno-unknown-pragmas

    INCLUDEPATH += /usr/include

    LIBS += -L/usr/local/lib \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc

    QMAKE_CXXFLAGS_WARN_ON = -Wno-unused-variable -Wno-reorder
}
