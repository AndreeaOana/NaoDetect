//
// Created by Barış Özcan on 07/01/16.
//

#include "Evaluator.h"
#include "SVMAnalysis.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

Evaluator::Evaluator(cv::Mat Response, cv::Mat GroundTruth,std::string FileName, ImageReader *readImages) {

    cv::Mat truepositives;
    cv::Mat falsepositives;
    cv::Mat falsenegatives;

    int tp;
    int fp = 0;
    int fn = 0;
    Response.convertTo(Response,CV_32S);
    GroundTruth.convertTo(GroundTruth,CV_32S);
    int cnt;
    FileName = "ConfusionMat"+FileName+".jpg";
    cv::Mat temp;
    cv::Mat ConfusionMatrixT = cv::Mat::zeros(2,2,CV_32S); // 15,15
    cv::Mat ConfusionMatrix ;

    this->readImages = readImages;

    for (int i = 0; i < Response.rows ; ++i) {

        cnt = ConfusionMatrixT.at<int>(Response.at<int>(i),GroundTruth.at<int>(i));
        ConfusionMatrixT.at<int>(Response.at<int>(i),GroundTruth.at<int>(i)) = cnt+1;
    }



    for (int j = 0; j < ConfusionMatrixT.rows ; ++j) {

        for (int i = 0; i <ConfusionMatrixT.cols ; ++i) {

            if (i==j){
                tp = ConfusionMatrixT.at<int>(j,i);
                truepositives.push_back(tp);
            } else  {

                fp += ConfusionMatrixT.at<int>(j,i);
                fn += ConfusionMatrixT.at<int>(i,j);
            }
        }
        falsepositives.push_back(fp);
        falsenegatives.push_back(fn);
        fp = 0;
        fn = 0;
    }


    std::cout<< " TRUE POSITIVES : " << truepositives.t()<<std::endl;
    std::cout<< " FALSE POSITIVES : " << falsepositives.t()<<std::endl;
    std::cout<< " FALSE NEGATIVES : " << falsenegatives.t()<<std::endl;

    cv::normalize(ConfusionMatrixT,ConfusionMatrix,0,255,CV_MINMAX,CV_32FC1);

//    cv::imwrite(FileName,ConfusionMatrix);
    Evaluator::computeAccuracy(Response,GroundTruth);

    cv::Mat SVM_ConfusionMat;
    computeConfusionMat(GroundTruth,Response,SVM_ConfusionMat);
//    std::cout<<SVM_ConfusionMat<<std::endl;
    cv::namedWindow("SVM confusion mat",CV_WINDOW_NORMAL);
    cv::imshow("SVM confusion mat",SVM_ConfusionMat);
    cv::waitKey(0);

}

void Evaluator::computeConfusionMat(cv::Mat &expected, cv::Mat &predicted, cv::Mat &confMat)
{
   // confMat = cv::Mat::zeros(15,15, CV_32FC1);  
   confMat = cv::Mat::zeros(2,2, CV_32FC1); 

    std::cout<<expected.rows<<" "<<expected.cols<<std::endl;
    for (int i = 0; i < expected.rows; i++) {
        confMat.at<float>(expected.at<int>(i,0),predicted.at<int>(i,0))++;
    }

   //std::cout<<expected.t()<<std::endl;
    //std::cout<<predicted.t()<<std::endl;

    for (int i = 0; i < confMat.rows; i++)
        confMat.row(i) /= sum(confMat.row(i))[0];
}

void Evaluator::computeAccuracy(cv::Mat Response, cv::Mat GroundTruth)
{

    // 5 NUM_OF_QUADRANTS + 1 (one is whole image itself)
    std::cout<<"\n computeAccuracy \n"<<std::endl;
    int numOfTestInputs = GroundTruth.rows/(1 + NUM_OF_QUADRANTS);
    cv::Mat GroundTruthLabels = GroundTruth(cv::Range(0,numOfTestInputs),cv::Range::all());
    std::cerr << "GroundTruthLabels: " << GroundTruthLabels.rows << " " << GroundTruthLabels.cols << std::endl;
    cv::Mat ResponseLabels;

    for (int i = 0; i < numOfTestInputs; i++) {
      //  int count[15] = {};
      //  int countFourquad[15] = {};
         int count[2] = {};
         int countFourquad[2] = {};
         for (int j = i; j < Response.rows; j += numOfTestInputs) {
            count[Response.at<int>(j)]++;

            if(j >= 17*numOfTestInputs)
            countFourquad[Response.at<int>(j)]++;
         }



        // major vote count
        int max = count[0];
        int index = 0;

        for (int j = 1; j < 2; j++) {  //<15
            if(max < count[j])
            {
                max = count[j];
                index = j;
            }
        }

        int theLabel = 0;

        // checkpoint
        if(max < 7)
        {
            ResponseLabels.push_back(Response.at<int>(i,0));
            theLabel=Response.at<int>(i,0);
        }
        else
        {
            ResponseLabels.push_back(index);
            theLabel=index;
        }

        std::cout<<"theLabel= "<<theLabel<<std::endl;        


        //salvat img in folderul theLabel
        if (theLabel==0) {
            std::stringstream ss2;
            ss2 << "../predictie0/robo_";
            ss2 << i;
            ss2 << ".png";
            std::string name = ss2.str();
            std::cout <<name<<std::endl;
            imwrite(name, this->readImages->Test_Images[i]);
        } 
        else if (theLabel!=0) {
            std::stringstream ss2;
            ss2 << "../predictie1/robo_";
            ss2 << i;
            ss2 << ".png";
            std::string name = ss2.str();
            std::cout<<name<<std::endl;

        }

//        if(max < 8)
//        {
//            // major vote count
//            int max = countFourquad[0];
//            int index = 0;

//            for (int j = 1; j < 15; j++) {
//                if(max < countFourquad[j])
//                {
//                    max = countFourquad[j];
//                    index = j;
//                }
//            }

//            if(max < 3)
//                ResponseLabels.push_back(Response.at<int>(i,0));
//            else
//                ResponseLabels.push_back(index);
//        }
//        else
//            ResponseLabels.push_back(index);

    }

    ResponseLabels.convertTo(ResponseLabels, CV_32S);
    GroundTruthLabels.convertTo(GroundTruthLabels, CV_32S);

    // for (int i=0; i<ResponseLabels.rows; ++i)
    // {
    //     std::cerr << "predicted " << ResponseLabels.at<int>(i,0) << ", real " << GroundTruthLabels.at<int>(i,0) << std::endl;
    // }

    // create 0,1 Mat (0: false, 1: true)
    cv::Mat out = (ResponseLabels == GroundTruthLabels)/255;
    std::cerr << out.rows << " " << out.cols << std::endl;
    // calculate accuracy
    this->accuracy = sum(out)[0]/out.rows;

    std::cout<< " ACCURACY : " << 100*this->accuracy <<"%"<< std::endl;
    std::cout<< " ACCURACY : " << ResponseLabels.rows << "  "<< numOfTestInputs<<std::endl;

}

Evaluator::~Evaluator() {

}
