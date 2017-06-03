/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek	S008284 Department of Electrical and Electronics
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
    std::string name;
    // std::vector<cv::Mat> testImages;
    std::vector<cv::Mat> testImages;

    ~SVMAnalysis();

private:

    //cv::Ptr<cv::ml::SVM> svm_obj;
    std::vector<cv::Ptr<cv::ml::SVM> > svm_objects;
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
    BagOfSIFT *bag;
    int getMinIndex(cv::Mat1f mat1);
    cv::Mat SVMDataCreate1vsALL(int nClass); //changes labels according to given range considering 1vsALL configuration
    void setSVMParams(cv::Ptr<cv::ml::SVM>&, const cv::Mat& responses, bool balanceClasses );
    void setSVMTrainAutoParams( cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid,
                                cv::ml::ParamGrid &p_grid, cv::ml::ParamGrid &nu_grid,
                                cv::ml::ParamGrid &coef_grid, cv::ml::ParamGrid &degree_grid );
    void SVMTester(cv::Ptr<cv::ml::SVM>&);
    void Evaluation(); 
    //void EvaluationRoi();
    int EvaluationPic(cv::Mat1f dataTestDescriptor);
    void SVMTrainer();
    cv::Ptr<cv::ml::SVM> CalculateSVM(cv::Mat label, std::string FileName);
};


#endif //PRJ3_008284_010097_SVMANALYSIS_H
