/*
  Muhammet Pakyürek	S008284 Department of Electrical and Electronics
  Baris Ozcan	S010097	Department of Computer Science
*/

#include "TinyImages.h"
#include<iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ImageReader.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#define K 20

TinyImages::TinyImages(ImageReader *DataSet) {



    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------TINY IMAGES RUNNING----------"<<std::endl;
    std::cout<<"     "<<std::endl;


    this->TestImages = DataSet->Test_Images;
    this->TrainImages = DataSet->Train_Images;
    this->TestLabels = DataSet->Test_Labels;
    this->TrainLabels = DataSet->Train_Labels;
    this->normalizedTestImages = this->TestImages;
    this->normalizedTrainImages = this->TrainImages;
    TinyImages::normalizeData();
   // TinyImages::CreateTinyImages();



    std::cout<<"no of test images :"<<this->TestImages.size()<<std::endl;
    std::cout<<"no of training images :"<<this->TrainImages.size()<<std::endl;
    std::cout<<"size of container :"<<trainDatavector.rows<<"x"<<testDatavector.cols<<std::endl;
    TinyImages::TinyImageTest();

}

void TinyImages::normalizeData() {

    cv::Mat resizedImage;
    cv::Size size(16,16); // size of tiny images.
    cv::Scalar mean;
    cv::Mat normalizedImage;

    for (int i = 0; i < this->TrainImages.size(); ++i) {

        mean = cv::mean(this->TrainImages[i]);
        cv::subtract(this->TrainImages[i],mean,normalizedImage);
        cv::resize(normalizedImage,resizedImage,size);
        resizedImage.convertTo(resizedImage,CV_32FC1);
        resizedImage.reshape(0,1).copyTo(resizedImage);
        resizedImage /= cv::norm(resizedImage);





        this->trainDatavector.push_back(resizedImage);

        //this->normalizedTrainImages[i] = normalizedImage;

    }



    for (int i = 0; i < this->TestImages.size(); ++i) {

        mean = cv::mean(this->TestImages[i]);
        cv::subtract(this->TestImages[i],mean,normalizedImage);
        cv::resize(normalizedImage,resizedImage,size);
        resizedImage.convertTo(resizedImage,CV_32FC1);
        resizedImage.reshape(0,1).copyTo(resizedImage);
        resizedImage /= cv::norm(resizedImage);
        this->testDatavector.push_back(resizedImage);

    }

}



void TinyImages::TinyImageTest(){


    float count = 0; //accurate guess count

    kn = cv::ml::KNearest::create();
    kn->train(this->trainDatavector,cv::ml::ROW_SAMPLE,this->TrainLabels);
    cv::Mat results = cv::Mat::zeros(1, K, CV_32FC1);




    for (int i = 0; i <testDatavector.rows ; ++i) {

        float response = kn->findNearest(testDatavector.row(i),K,results);
        if (response  == this->TestLabels.at<int>(i) ) {
            count++;
        }

    }
    float accuracy = (count/testDatavector.rows)*100;
    std::cout<<accuracy<< "% accuracy"<<std::endl;


}



TinyImages::~TinyImages() {

}
