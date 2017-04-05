/*
Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science
*/

#include "KNNTest.h"
#include "BagOfSIFT.h"
#include<iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "ImageReader.h"
#include "Evaluator.h"

#define K 50


KNNTest::KNNTest(BagOfSIFT *BagOfSIFT) {

    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------KNNTest RUNNING----------"<<std::endl;
    std::cout<<"     "<<std::endl;


//    this->dataTestDescriptor = BagOfSIFT->dataTestDescriptor;
//    this->dataTrainDescriptor = BagOfSIFT->dataTrainDescriptor;
//    this->trainLabels = BagOfSIFT->TrainLabels;
//    this->testLabels = BagOfSIFT->TestLabels;

    cv::Mat tmpDataTestDescriptor = BagOfSIFT->dataTestDescriptor;
    cv::Mat tmpDataTrainDescriptor = BagOfSIFT->dataTrainDescriptor;
    cv::Mat tmpTrainLabels = BagOfSIFT->TrainLabels;
    cv::Mat tmpTestLabels = BagOfSIFT->TestLabels;

    int trainWholeImgPart = tmpDataTrainDescriptor.rows/17;
    int testWholeImgPart = tmpDataTestDescriptor.rows/17;
    int startTrn = 0;//trainWholeImgPart;
    int startTst = 0;//testWholeImgPart;

    std::cout<<"trainWholeImgPart "<<trainWholeImgPart<<std::endl;
    std::cout<<"trainLabels "<<BagOfSIFT->dataTestDescriptor.rows<<std::endl;


    this->dataTestDescriptor = tmpDataTestDescriptor(cv::Range(startTst,startTst + testWholeImgPart),cv::Range::all());
    this->dataTrainDescriptor = tmpDataTrainDescriptor(cv::Range(startTrn,startTrn + trainWholeImgPart),cv::Range::all());
    this->trainLabels = tmpTrainLabels(cv::Range(startTrn,startTrn + trainWholeImgPart),cv::Range::all());
    this->testLabels = tmpTestLabels(cv::Range(startTst,startTst + testWholeImgPart),cv::Range::all());


    cv::Mat responses;
    std::cout<<"KNN Training Descriptor Size : "<<this->dataTrainDescriptor.rows<<"x"<<this->dataTrainDescriptor.cols<<std::endl;
    std::cout<<"KNN Descriptor Size : "<<this->dataTestDescriptor.rows<<"x"<<this->dataTestDescriptor.cols<<std::endl;
    float count = 0; //accurate guess count
    kn = cv::ml::KNearest::create();
    kn->train(this->dataTrainDescriptor,cv::ml::ROW_SAMPLE,this->trainLabels);
    cv::Mat results;

    for (int i = 0; i <this->dataTestDescriptor.rows ; ++i) {

        float response = kn->findNearest(this->dataTestDescriptor.row(i),K,results);
        responses.push_back(response);
        if (response  == this->testLabels.at<int>(i) ) {
            count++;
        }
    }
    Evaluator Evaluator(responses,this->testLabels,"KNN");

    float accuracy = (count/this->dataTestDescriptor.rows)*100;
    std::cout<<"Bag Of SIFT has "<< accuracy << "% accuracy."<<std::endl;
}

KNNTest::~KNNTest() {

}
