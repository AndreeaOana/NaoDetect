/*
Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science
*/

#ifndef PRJ3_008284_010097_SVMANALYSIS_H
#define PRJ3_008284_010097_SVMANALYSIS_H

#include<iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ImageReader.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "BagOfSIFT.h"

class SVMAnalysis {

public:

    SVMAnalysis(BagOfSIFT *BagOfSIFT, ImageReader *readImages);

    ~SVMAnalysis();

private:

    cv::Ptr<cv::ml::SVM> svm_obj;
    cv::Mat1f Total_Response; // Rows : Label number, Columns : confidence for each test
    cv::Ptr<cv::ml::TrainData> TrainData;
    cv::Mat1f dataTrainDescriptor;
    cv::Mat1f dataTestDescriptor;
    cv::Mat trainLabels;
    cv::Mat testLabels;
    cv::Mat1i Consensus; // major vote container

    ImageReader *readImages;
    void saveImages();


    // NM
    int C_value = 90;
    int getMinIndex(cv::Mat1f mat1);
    cv::Mat SVMDataCreate1vsALL(int nClass); //changes labels according to given range considering 1vsALL configuration
    void setSVMParams(const cv::Mat& responses, bool balanceClasses );
    void setSVMTrainAutoParams( cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid,
                                cv::ml::ParamGrid &p_grid, cv::ml::ParamGrid &nu_grid,
                                cv::ml::ParamGrid &coef_grid, cv::ml::ParamGrid &degree_grid );
    void SVMTester();
    void Evaluation();
    void SVMTrainer();
    void CalculateSVM(cv::Mat label, std::string FileName);
};


#endif //PRJ3_008284_010097_SVMANALYSIS_H
