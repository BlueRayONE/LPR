/*
@InProceedings{WolfICPR2002V,
  Author         = {C. Wolf and J.-M. Jolion and F. Chassaing},
  Title          = {Text {L}ocalization, {E}nhancement and {B}inarization in {M}ultimedia {D}ocuments},
  BookTitle      = {Proceedings of the {I}nternational {C}onference on {P}attern {R}ecognition},
  Volume         = {2},
  Pages          = {1037-1040},
  year           = 2002,
}
*/

#ifndef BINARIZEWOLFJOLION_H
#define BINARIZEWOLFJOLION_H

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

enum NiblackVersion
{
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

#define BINARIZEWOLF_VERSION	"2.4 (August 1st, 2014)"

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;

static void usage(char *com);
double calcLocalStats(cv::Mat& im, cv::Mat& map_m, cv::Mat& map_s, int winx, int winy);
void NiblackSauvolaWolfJolion(cv::Mat im, cv::Mat output, NiblackVersion version,int winx, int winy, double k, double dR);


#endif // BINARIZEWOLFJOLION_H
