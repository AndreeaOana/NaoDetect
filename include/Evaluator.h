//
// Created by Barış Özcan on 07/01/16.
//

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
