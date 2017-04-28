/*
Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science
*/

#include "SVMAnalysis.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <sstream> 
#include <vector>
#include<dirent.h>
#include <opencv2/core/core.hpp>



//#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ImageReader.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "BagOfSIFT.h"
#include "Evaluator.h"

#define NUM_OF_CLASS 2

using namespace cv;


SVMAnalysis::SVMAnalysis(BagOfSIFT *BagOfSIFT, ImageReader *readImages) {

    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------SVM Analysis RUNNING----------"<<std::endl;
    std::cout<<"     "<<std::endl;


    Mat LoadedImage;

    int windows_n_rows = 60;
    int windows_n_cols = 60;
    int StepSlide = 30;


    this->readImages = readImages;

    this->bag = BagOfSIFT;

    cv::Mat tmpDataTestDescriptor = BagOfSIFT->dataTestDescriptor;
    cv::Mat tmpDataTrainDescriptor = BagOfSIFT->dataTrainDescriptor;
    cv::Mat tmpTrainLabels = BagOfSIFT->TrainLabels;
    cv::Mat tmpTestLabels = BagOfSIFT->TestLabels;


     std::vector<cv::Mat> testImages = readImages->Test_Images;


    tmpTrainLabels = tmpTrainLabels(cv::Range(0,tmpDataTrainDescriptor.rows),cv::Range::all());
    tmpTestLabels = tmpTestLabels(cv::Range(0,tmpDataTestDescriptor.rows),cv::Range::all());

    std::cout<<"\n tmpDataTestDescriptor: "<<tmpDataTestDescriptor.rows<<std::endl;
    std::cout<<"tmpDataTrainDescriptor: "<<tmpDataTrainDescriptor.rows<<std::endl;
    std::cout<<"tmpTrainLabels: "<<tmpTrainLabels.rows<<std::endl;
    std::cout<<"tmpTestLabels: "<<tmpTestLabels.rows<<std::endl;

    std::cout<<std::endl;

//    cv::Mat tmpDescriptor;
//    cv::Mat tmpLabels;

//    cv::FileStorage fsReader("Train_300_20_4.yml",cv::FileStorage::READ);
//    if(!fsReader.isOpened())
//        std::cout<<"file could not be opened\n";
//    fsReader["descriptors"] >> tmpDescriptor;
//    fsReader["labels"] >> tmpLabels;
//    fsReader.release();

//    std::cout<<"\n Before: "<<std::endl;
//    std::cout<<"\n tmpLabels: "<<tmpLabels.rows<<std::endl;
//    std::cout<<"\n tmpDescriptor: "<<tmpDescriptor.rows<<std::endl;

//    int rowCount = tmpDescriptor.rows;
//    tmpLabels = tmpLabels(cv::Range(rowCount/5,rowCount),cv::Range::all());
//    tmpDescriptor = tmpDescriptor(cv::Range(rowCount/5,rowCount),cv::Range::all());

//    std::cout<<"\n After: "<<std::endl;
//    std::cout<<"\n tmpLabels: "<<tmpLabels.rows<<std::endl;
//    std::cout<<"\n tmpDescriptor: "<<tmpDescriptor.rows<<std::endl;


//    tmpDataTrainDescriptor.push_back(tmpDescriptor);
//    tmpTrainLabels.push_back(tmpLabels);

//    cv::FileStorage fsWriter("Train_300_20_20.yml",cv::FileStorage::WRITE);
//    if(!fsWriter.isOpened())
//        std::cout<<"file could not be opened\n";
//    fsWriter<< "descriptors" << tmpDataTrainDescriptor;
//    fsWriter<< "labels" << tmpTrainLabels;
//    //release the file storage
//    fsWriter.release();

//    std::cout<<"Success!\n";
//    return;


    int trainWholeImgPart = tmpDataTrainDescriptor.rows/(1 + NUM_OF_QUADRANTS);
    int testWholeImgPart = tmpDataTestDescriptor.rows/(1 + NUM_OF_QUADRANTS);

    std::cout<<"testWholeImgPart : "<<testWholeImgPart<<" "<<tmpTestLabels.rows<<std::endl;

    for (int i = 0; i < (1 + NUM_OF_QUADRANTS); i++) {
        this->C_value = 100 + i*20;
        if(i > 2)   //15
            this->C_value = 110; //140
    //    else if(i > 0)
    //        this->C_value = 500;
        int startTrn = i*trainWholeImgPart;
        int startTst = i*testWholeImgPart;

        this->dataTestDescriptor = tmpDataTestDescriptor(cv::Range(startTst,startTst + testWholeImgPart),cv::Range::all());
        this->dataTrainDescriptor = tmpDataTrainDescriptor(cv::Range(startTrn,startTrn + trainWholeImgPart),cv::Range::all());
        this->trainLabels = tmpTrainLabels(cv::Range(startTrn,startTrn + trainWholeImgPart),cv::Range::all());
        this->testLabels = tmpTestLabels(cv::Range(startTst,startTst + testWholeImgPart),cv::Range::all());


        //Training Data type check
        int nType = this->dataTrainDescriptor.depth();
        if(nType != CV_32F)
        {
            dataTrainDescriptor.convertTo(dataTrainDescriptor, CV_32F);
            dataTestDescriptor.convertTo(dataTestDescriptor, CV_32F);
        }

        nType = this->trainLabels.depth();
        if(nType != CV_32S)
        {
            this->trainLabels.convertTo(this->trainLabels, CV_32S);
            this->trainLabels.convertTo(this->trainLabels, CV_32S);
        }

        SVMAnalysis::SVMTrainer();

       
         // imwrite(name, this->readImages->Test_Images[i]);

       
   //     LoadedImage = imread("../predictie/frame_%06d.jpg", IMREAD_COLOR);

        // cv::String path("../predictie/frame_%06d.jpg"); //select only jpg
        // std::vector<String> fn;
        // std::vector<cv::Mat> data;
        // cv::glob(path,fn,true); // recurse
        // for (size_t k=0; k<fn.size(); ++k) {
        //     cv::Mat LoadedImage = cv::imread(fn[k]);
        //   //  if (LoadedImage.empty()) continue; //only proceed if sucsessful
        // data.push_back(LoadedImage);
        // }


        // Mat DrawResultGrid = LoadedImage.clone();

        //     for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide) {
        //          for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide) {

        //              namedWindow("Step 1 image loaded", WINDOW_AUTOSIZE);
        //             imshow("Step 1 image loaded", LoadedImage);
        //             waitKey(1000);


        //             cv::Rect windows(col, row, windows_n_rows, windows_n_cols);
   
        //             cv::Mat DrawResultHere =  LoadedImage.clone();

        //             rectangle(DrawResultHere, windows, Scalar(255), 1, 8, 0);  //pt dreptunghi
        //             rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);  //si grid
 
                

        //             namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
        //             imshow("Step 2 draw Rectangle", DrawResultHere);
        //             waitKey(100);
        //             imwrite("Step2.JPG", DrawResultHere);

        //             namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
        //             imshow("Step 3 Show Grid", DrawResultGrid);
        //             waitKey(100);
        //             imwrite("Step3.JPG", DrawResultGrid);

        //             cv::Mat Roi = LoadedImage(windows);

        //             namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
        //             imshow("Step 4 Draw selected Roi", Roi);
        //             waitKey(100);
        //             imwrite("Step4.JPG", Roi);

        //             // SVMAnalysis::EvaluationRoi();

        //         }
        //     }

        SVMAnalysis::Evaluation();
    //   SVMAnalysis::saveImages();

    //     if (i == NUM_OF_QUADRANTS) {
    //     SVMAnalysis::saveImages();
    // }

        this->Total_Response.release();
        std::cout<<"C_value= "<<C_value<<std::endl;
    }

    std::vector<cv::Mat> pics = this->readImages->Test_Images;

    for (int i=0;i<pics.size();i++) {

        EvaluationPic(bag->Extract_BOF_testing(pics[i]));
    }

    Evaluator evaluator(this->Consensus,tmpTestLabels,"SVM", readImages);
//    std::cout<<"classification rate : "<<evaluator.getAccuracy()<< "%"<<std::endl;
}

void SVMAnalysis::saveImages() {

    std::cout<<"SAVE IMAGES"<<std::endl;
 //   std::cout<<"total response"<<this->Total_Response.rows<<"x"<<this->Total_Response.cols <<std::endl;

    int consensus;

    //this->Total_Response = this->Total_Response.t();
    //std::cout<<this->Total_Response.rows<<"x"<<this->Total_Response.cols <<std::endl;
    //std::cout<<"total response"<<Total_Response<<std::endl;

    for (int i = 0; i <this->Total_Response.rows ; ++i) {
        
     consensus = getMinIndex(this->Total_Response.row(i));
//predictia - rezultatul pt imag i din test

       std::cout<<consensus<<std::endl;
        if (consensus==0) {
            std::stringstream ss2;
            ss2 << "../predictie0/robo_";
            ss2 << i;
            ss2 << ".png";
            std::string name = ss2.str();
            std::cout <<name<<std::endl;
            imwrite(name, this->readImages->Test_Images[i]);
        } 
        else if (consensus==1) {
            std::stringstream ss2;
            ss2 << "../predictie1/robo_";
            ss2 << i;
            ss2 << ".png";
            std::string name = ss2.str();
            std::cout<<name<<std::endl;
        }

    }

}

void SVMAnalysis::EvaluationRoi() {
    std::cout<<"SVMAnalysis::EvaluationRoi"<<std::endl;
    float count=0;
    int consensus;
     
    this->Total_Response = this->Total_Response.t();
    //std::cout<<"Total_Response.rows "<<this->Total_Response.rows<<" x "<<" Total_Response.cols "<<this->Total_Response.cols <<std::endl;

    //for (int i = 0; i <this->Total_Response.rows ; ++i) {
        
        consensus = getMinIndex(this->Total_Response);
//predictia - rezultatul pt imag i din test
        //if(consensus == this->testLabels.at<int>(i))
          //  count++;
        this->Consensus.push_back(consensus);
        std::cout<<"consensus "<<consensus<<std::endl;

     
   // }

    std::cout<<"testLabels.rows= "<<testLabels.rows<<std::endl;
    std::cout<<"counter classification rate : "<<count/this->testLabels.rows*100<< "%"<<std::endl;
//    Evaluator evaluator(this->Consensus,this->testLabels,"SVM");
//    std::cout<<"classification rate : "<<evaluator.getAccuracy()<< "%"<<std::endl;

//    this->allQuadrabtsResponse.push_back();
    std::cout<<"\n Consensus: "<<this->Consensus.rows<<"\n"<<std::endl;

}

void SVMAnalysis::Evaluation() {
    std::cout<<"SVMAnalysis::Evaluation"<<std::endl;
    float count=0;
    int consensus;
     
    this->Total_Response = this->Total_Response.t();
    std::cout<<"Total_Response.rows "<<this->Total_Response.rows<<" x "<<" Total_Response.cols "<<this->Total_Response.cols <<std::endl;

    for (int i = 0; i <this->Total_Response.rows ; ++i) {
        
        consensus = getMinIndex(this->Total_Response.row(i));
//predictia - rezultatul pt imag i din test
        if(consensus == this->testLabels.at<int>(i))
            count++;
        this->Consensus.push_back(consensus);
        //std::cout<<"consensus "<<consensus<<std::endl;

     
    }

    std::cout<<"testLabels.rows= "<<testLabels.rows<<std::endl;
    std::cout<<"counter classification rate : "<<count/this->testLabels.rows*100<< "%"<<std::endl;
//    Evaluator evaluator(this->Consensus,this->testLabels,"SVM");
//    std::cout<<"classification rate : "<<evaluator.getAccuracy()<< "%"<<std::endl;

//    this->allQuadrabtsResponse.push_back();
    std::cout<<"\n Consensus: "<<this->Consensus.rows<<"\n"<<std::endl;

}

int SVMAnalysis::EvaluationPic(cv::Mat1f dataTestDescriptor) {
    std::cout<<"SVMAnalysis::EvaluationRoi"<<std::endl;
    float count=0;
    int consensus;
     
    //this->Total_Response = this->Total_Response.t();
    //std::cout<<"Total_Response.rows "<<this->Total_Response.rows<<" x "<<" Total_Response.cols "<<this->Total_Response.cols <<std::endl;
       
 //   BagOfSIFT::Extract_BOF_testing();
   // image = BagOfSIFT::Extract_BOF_testing(this->dataTestDescriptor);

        
    // consensus = getMinIndex(this->Total_Response.row(i));
//predictia - rezultatul pt imag i din test

       std::cout<<consensus<<std::endl;
    

        
  
   // this->Consensus.push_back(consensus);
    std::cout<<"consensus "<<consensus<<std::endl;

     
   // }

    std::cout<<"testLabels.rows= "<<testLabels.rows<<std::endl;
    std::cout<<"counter classification rate : "<<count/this->testLabels.rows*100<< "%"<<std::endl;
//    Evaluator evaluator(this->Consensus,this->testLabels,"SVM");
//    std::cout<<"classification rate : "<<evaluator.getAccuracy()<< "%"<<std::endl;

//    this->allQuadrabtsResponse.push_back();
    std::cout<<"\n Consensus: "<<this->Consensus.rows<<"\n"<<std::endl;

}


int SVMAnalysis::getMinIndex(cv::Mat1f mat1) {

    if (mat1.rows != 1)
        std::invalid_argument("");

    int minIndex = 0;
    float minValue = mat1.at<float>(0);
    for (int i = 1; i < mat1.cols; ++i) {
        float value = mat1.at<float>(i);
        if (value < minValue) {
            minIndex = i;
            minValue = value;
           // std::cout<<"MINVAL : " <<minIndex<<std::endl;
        }
    }

    return minIndex;

}



void SVMAnalysis::SVMTester(cv::Ptr<cv::ml::SVM>& svm_obj) {

    cv::Mat1f temp;
    cv::Mat1f resultsMat;
    svm_obj->predict(this->dataTestDescriptor,resultsMat,cv::ml::StatModel::RAW_OUTPUT);

    temp = resultsMat.t();
    this->Total_Response.push_back(temp);
}


void SVMAnalysis::SVMTrainer() {
    std::cout<<"SVMAnalysis::SVMTrainer()"<<std::endl;
    std::string labelstring;
    cv::Mat1i currentSVMlabels; // temporal label container for 15 SVM's

    for(int j=0; j<NUM_OF_CLASS;j++) {

        labelstring= std::to_string(j);
        std::string FileName("SVM_train_"+ labelstring);
        currentSVMlabels= SVMAnalysis::SVMDataCreate1vsALL(j);
        cv::Ptr<cv::ml::SVM> svm_obj = CalculateSVM(currentSVMlabels, FileName);
        this->svm_objects.push_back(svm_obj);
        SVMAnalysis::SVMTester(svm_obj);

        std::cout<<"C Value : "<< svm_obj->getC()  <<std::endl;
    }
    

}


cv::Ptr<cv::ml::SVM> SVMAnalysis::CalculateSVM(cv::Mat label, std::string FileName) {

    cv::Ptr<cv::ml::SVM> svm_obj;
    svm_obj = cv::ml::SVM::create();
    svm_obj->setC(this->C_value);
    setSVMParams(svm_obj, label, true );
    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(this->dataTrainDescriptor, cv::ml::ROW_SAMPLE, label);
    svm_obj->train(tData);
//    this->svm_obj->save(FileName);
    return svm_obj;
}

void SVMAnalysis::setSVMParams(cv::Ptr<cv::ml::SVM>& svm_obj, const cv::Mat &responses, bool balanceClasses) {

    int pos_ex = countNonZero(responses == 1);
    int neg_ex = countNonZero(responses == -1);
    svm_obj->setType(cv::ml::SVM::C_SVC);
    svm_obj->setKernel(cv::ml::SVM::LINEAR);
    if( balanceClasses )
    {
        cv::Mat class_wts( 2, 1, CV_32FC1 );
        // The first training sample determines the '+1' class internally, even if it is negative,
        // so store whether this is the case so that the class weights can be reversed accordingly.
        bool reversed_classes = (responses.at<int>(0) < 0.f);
        if( reversed_classes == false )
        {
            class_wts.at<float>(0) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
            class_wts.at<float>(1) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
        }
        else
        {
            class_wts.at<float>(0) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex);
            class_wts.at<float>(1) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex);
            float temp = class_wts.at<float>(0);
            class_wts.at<float>(0) = class_wts.at<float>(1);
            class_wts.at<float>(1) = temp;
        }
        svm_obj->setClassWeights(class_wts);
    }
}


void SVMAnalysis::setSVMTrainAutoParams( cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid,
        cv::ml::ParamGrid &p_grid, cv::ml::ParamGrid &nu_grid,
        cv::ml::ParamGrid &coef_grid, cv::ml::ParamGrid &degree_grid ){
}

cv::Mat SVMAnalysis::SVMDataCreate1vsALL( int nClass) {

    cv::Mat SVM1vsAllTrainLabel = cv::Mat::ones(this->trainLabels.rows,this->trainLabels.cols,CV_32S)*-1;
    for(int i=0; i<this->trainLabels.rows;i++)
        if(this->trainLabels.at<int>(i) == nClass)
            SVM1vsAllTrainLabel.at<int>(i) = 1;
    return SVM1vsAllTrainLabel;
}

SVMAnalysis::~SVMAnalysis() {

}
