/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science

*/
#ifndef PRJ3_008284_010097_EVALUATOR_H
#define PRJ3_008284_010097_EVALUATOR_H


#include <opencv2/core/mat.hpp>
#include "ImageReader.h"

class Evaluator {

public:
    Evaluator(cv::Mat Response, cv::Mat GroundTruth, std::string FileName, ImageReader *readImages);
    inline const float getAccuracy(){return accuracy;}

    ~Evaluator();

private:
	ImageReader *readImages;

    void computeAccuracy(cv::Mat Response, cv::Mat GroundTruth);
    void computeConfusionMat(cv::Mat &expected, cv::Mat &predicted, cv::Mat &confMat);
    float accuracy;
};


#endif //PRJ3_008284_010097_EVALUATOR_H
