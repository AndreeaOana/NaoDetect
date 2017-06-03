/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek   S008284 Department of Electrical and Electronics
        Baris Ozcan S010097 Department of Computer Science

*/

#include "BagOfSIFT.h"
#include<iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "ImageReader.h"

#include <chrono>


BagOfSIFT::BagOfSIFT(ImageReader *DataSet) {

    std::cout.flush();
    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------BAG OF SIFT CREATING----------"<<std::endl;
    std::cout<<"     "<<std::endl;


    this->TestImages = DataSet->Test_Images;
    this->TrainImages = DataSet->Train_Images;
    this->TestLabels = DataSet->Test_Labels;
    this->TrainLabels = DataSet->Train_Labels;

    // NM
    this->dataTestQuadrantDescriptor =  std::vector<cv::Mat1f>(NUM_OF_QUADRANTS);
    this->dataTrainQuadrantDescriptor =  std::vector<cv::Mat1f>(NUM_OF_QUADRANTS);



    this->dictionarysize = 200;  //300
    this->StepSize = 10;  //20
    this->StepSizeString = std::to_string(this->StepSize);
    this->DictionarySizeString = std::to_string(this->dictionarysize);
    this->FileName = "dictionary_" + this->DictionarySizeString +"_"+ this->StepSizeString + ".yml";
    cv::Mat Label; // Label Dump

    this->TestFileName = "Test_"+ this->DictionarySizeString +"_"+ this->StepSizeString + "_" + std::to_string(NUM_OF_QUADRANTS) + ".yml";
    this->TrainFileName = "Train_"+ this->DictionarySizeString +"_"+ this->StepSizeString + "_" + std::to_string(NUM_OF_QUADRANTS) +".yml";

    std::cout<<"creating"<<this->FileName<<"\n "<<this->TestFileName<<"\n "<<this->TrainFileName <<std::endl;

    BagOfSIFT::loadDataFile(this->TrainFileName,this->dataTrainDescriptor,this->TrainLabels);
    BagOfSIFT::loadDataFile(this->TestFileName,this->dataTestDescriptor,this->TestLabels);

    if(NUM_OF_QUADRANTS == 1)
        for (int i = 0; i < NUM_OF_QUADRANTS + 1; i++) {
            this->TestLabels.push_back(DataSet->Test_Labels);
            this->TrainLabels.push_back(DataSet->Train_Labels);
        }


    //Store the vocabulary
    cv::FileStorage fs (this->FileName,cv::FileStorage::READ);
    fs["vocabulary"] >>this->dictionary;
    fs.release();

    if(!this->dataTestDescriptor.data || !this->dataTestDescriptor.data) {
        std::cout<<"There is not a descriptor data... Creating.."<<std::endl;
        BagOfSIFT::Extract_BOF_features();
    }
    std::cout<<"Dictionary size : "<<this->dictionary.rows<<"x"<<this->dictionary.cols<<std::endl;
    std::cout<<"Training Descriptor Size : "<<this->dataTrainDescriptor.rows<<"x"<<this->dataTrainDescriptor.cols<<std::endl;
    std::cout<<"Test Descriptor Size : "<<this->dataTestDescriptor.rows<<"x"<<this->dataTestDescriptor.cols<<std::endl;
}

cv::Mat1f BagOfSIFT::Extract_BOF_testing(cv::Mat Img) {   


  //  cv::Mat Img; //temporal image container
    cv::Mat1f descriptor;
    //create SIFT feature extractor
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();

    cv::Ptr<cv::DescriptorMatcher> descMatcher(new cv::FlannBasedMatcher());

    //create SIFT descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> descExtractor = f2d;
    //Keypoint Storage
    std::vector<cv::KeyPoint> KeyPoints;
    cv::KeyPoint KeyPoint;
    //BoW storage
    cv::Mat BoWImageDescriptors;

      //create BoW descriptor exctractor
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE = cv::makePtr<cv::BOWImgDescriptorExtractor>( descExtractor,
                                                                                               descMatcher);

    //Set the dictionary with the vocabulary we created in the first step
    bowDE->setVocabulary(this->dictionary);


        std::cout<<"\n Extracting SIFT for TEST images"<<std::endl;


            
        //Img = dataTestDescriptor;

        imshow("imag",Img);
        cv::waitKey(10);
        //sanity check
        if(!Img.data)
        {
            std::cerr << "\r Problem loading image!!!" << std::endl;
           // return 0;
        }

        std::vector<std::vector<cv::KeyPoint>> quadrantKeyPoints(NUM_OF_QUADRANTS);
        int height = Img.rows;
        int width = Img.cols;

        for (int j = StepSize/2; j < height ; j+=StepSize/2) {
            for (int k = StepSize/2 ; k < width ; k+=StepSize/2) {
                KeyPoint = cv::KeyPoint(cv::Point2f(j,k),this->keypointsize);
                KeyPoints.push_back(KeyPoint);

                // NM add to corresponding quadrant
                quadrantKeyPoints[floor(std::sqrt(NUM_OF_QUADRANTS)*j/height)*std::sqrt(NUM_OF_QUADRANTS) + floor(std::sqrt(NUM_OF_QUADRANTS)*k/width)].push_back(KeyPoint);
            }

        }


        // NM compute descriptors
        for (int i = 0; i < NUM_OF_QUADRANTS; i++) {
            bowDE->compute(Img,quadrantKeyPoints[i],BoWImageDescriptors);
            quadrantKeyPoints[i].clear();

            // if you put this here, shouldn't you add label accordngly?
            if (BoWImageDescriptors.empty())
               // return 0;

            this->dataTestQuadrantDescriptor[i].push_back(BoWImageDescriptors);
        }


        auto T0 = std::chrono::high_resolution_clock::now();

        bowDE->compute(Img,KeyPoints,BoWImageDescriptors);

        auto T1 = std::chrono::high_resolution_clock::now();


        std::chrono::duration<float> diff1 = T1 - T0;
                  
        std::chrono::milliseconds d3 = std::chrono::duration_cast<std::chrono::milliseconds>(diff1);

        std::cout << "Feature extraction time: " << d3.count() << "ms\n";

                               

        KeyPoints.clear();
        //if (BoWImageDescriptors.empty())
           // return 0;
   //     this->dataTestDescriptor.push_back(BoWImageDescriptors);

        return BoWImageDescriptors;

        std::cout<<"BoWImageDescriptors "<<BoWImageDescriptors.size()<<std::endl;
}

void BagOfSIFT::Extract_BOF_features(){

    if(!this->dictionary.data){
        std::cout<<"\r NO DICTIONARY, creating..."<<std::endl;
        BagOfSIFT::BuildBOF();
    }

    cv::Mat Img; //temporal image container
    cv::Mat1f descriptor;
    //create SIFT feature extractor
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    //create a nearest neighbour matcher
    cv::Ptr<cv::DescriptorMatcher> descMatcher(new cv::FlannBasedMatcher());
    //create SIFT descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> descExtractor = f2d;
    //Keypoint Storage
    std::vector<cv::KeyPoint> KeyPoints;
    cv::KeyPoint KeyPoint;
    //BoW storage
    cv::Mat BoWImageDescriptors;


    if( !f2d || !descExtractor || !descMatcher)
    {
        std::cout << " \n featureDetector or descExtractor was not created" <<std::endl;
        return;
    }

    //create BoW descriptor exctractor
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE = cv::makePtr<cv::BOWImgDescriptorExtractor>( descExtractor,
                                                                                               descMatcher);

    //Set the dictionary with the vocabulary we created in the first step
    bowDE->setVocabulary(this->dictionary);
    //To store the image tag name - only for save the descriptor in a file
    char imageTag[10] = {"\0"};



    //open the file to write the resultant descriptor
    cv::FileStorage fstrain(this->TrainFileName, cv::FileStorage::WRITE);
    cv::FileStorage fstest(this->TestFileName,cv::FileStorage::WRITE);




    std::cout<<"Extracting SIFT for Training images"<<std::endl;
    //Extract SIFT features for TRAIN IMAGES
    for (int i = 0; i <this->TrainImages.size(); ++i){

        float percentage = (float)(i+1)/(float)this->TrainImages.size()*100;
        std::cout<<"\r Computing "<<percentage<<"% done" ;

        Img = this->TrainImages[i];
        //sanity check
        if(!Img.data)
        {
            std::cerr << "Problem loading image!!!" << std::endl;
            continue;
        }

        //  Here we have to seperate into 4 quadrants and extract descriptors for each quadrant
        //  We might collect Keypoints for each quadrant seperately in this for loop

        //  Keypoint Storage for each quadrant
        std::vector<std::vector<cv::KeyPoint>> quadrantKeyPoints(NUM_OF_QUADRANTS);
        int width = Img.cols;
        int height = Img.rows;

        for (int j = StepSize/2; j < height ; j+=StepSize/2) {
            for (int k = StepSize/2 ; k < width ; k+=StepSize/2) {
                KeyPoint = cv::KeyPoint(cv::Point2f(j,k),this->keypointsize);
                KeyPoints.push_back(KeyPoint);

                // NM add to corresponding quadrant
                quadrantKeyPoints[floor(std::sqrt(NUM_OF_QUADRANTS)*j/height)*std::sqrt(NUM_OF_QUADRANTS) + floor(std::sqrt(NUM_OF_QUADRANTS)*k/width)].push_back(KeyPoint);
            }
        }



        // NM compute descriptors
        for (int i = 0; i < NUM_OF_QUADRANTS; i++) {
            std::cout << "descriptor size: " << quadrantKeyPoints[i].size() << std::endl;

            bowDE->compute(Img,quadrantKeyPoints[i],BoWImageDescriptors);
            quadrantKeyPoints[i].clear();

            // if you put this here, shouldn't you add label accordngly?
            if (BoWImageDescriptors.empty())
                continue;

            this->dataTrainQuadrantDescriptor[i].push_back(BoWImageDescriptors);
        }

        bowDE->compute(Img,KeyPoints,BoWImageDescriptors);
        KeyPoints.clear();

        // if you put this here, shouldn't you add label accordngly?
        if (BoWImageDescriptors.empty())
            continue;

        this->dataTrainDescriptor.push_back(BoWImageDescriptors);
    }





    // std::cout<<"\n Extracting SIFT for TEST images"<<std::endl;
    // for (int i = 0; i <this->TestImages.size(); ++i){
    //     float percentage = (float)(i+1)/(float)this->TestImages.size()*100;
    //     std::cout<<"\r Computing "<<percentage<<"% done" ;
    //     Img = this->TestImages[i];
    //     //sanity check
    //     if(!Img.data)
    //     {
    //         std::cerr << "\r Problem loading image!!!" << std::endl;
    //         continue;
    //     }

    //     std::vector<std::vector<cv::KeyPoint>> quadrantKeyPoints(NUM_OF_QUADRANTS);
    //     int height = Img.rows;
    //     int width = Img.cols;

    //     for (int j = StepSize/2; j < height ; j+=StepSize/2) {
    //         for (int k = StepSize/2 ; k < width ; k+=StepSize/2) {
    //             KeyPoint = cv::KeyPoint(cv::Point2f(j,k),this->keypointsize);
    //             KeyPoints.push_back(KeyPoint);

    //             // NM add to corresponding quadrant
    //             quadrantKeyPoints[floor(std::sqrt(NUM_OF_QUADRANTS)*j/height)*std::sqrt(NUM_OF_QUADRANTS) + floor(std::sqrt(NUM_OF_QUADRANTS)*k/width)].push_back(KeyPoint);
    //         }

    //     }


    //     // NM compute descriptors
    //     for (int i = 0; i < NUM_OF_QUADRANTS; i++) {
    //         bowDE->compute(Img,quadrantKeyPoints[i],BoWImageDescriptors);
    //         quadrantKeyPoints[i].clear();

    //         // if you put this here, shouldn't you add label accordngly?
    //         if (BoWImageDescriptors.empty())
    //             continue;

    //         this->dataTestQuadrantDescriptor[i].push_back(BoWImageDescriptors);
    //     }

    //     std::cout<<"\r Computing "<<percentage<<"% done" ;
    //     bowDE->compute(Img,KeyPoints,BoWImageDescriptors);
    //     KeyPoints.clear();
    //     if (BoWImageDescriptors.empty())
    //         continue;
    //     this->dataTestDescriptor.push_back(BoWImageDescriptors);
    // }


    //Save descriptors
    cv::Mat tmpTrainLabels = this->TrainLabels;
   // cv::Mat tmpTestLabel = this->TestLabels;
    for (int i = 0; i < NUM_OF_QUADRANTS; i++) {
    //    this->dataTestDescriptor.push_back(this->dataTestQuadrantDescriptor[i]);
     //   this->TestLabels.push_back(tmpTestLabel);

        this->dataTrainDescriptor.push_back(this->dataTrainQuadrantDescriptor[i]);
        this->TrainLabels.push_back(tmpTrainLabels);
    }

    saveDataFile(this->TrainFileName,dataTrainDescriptor,this->TrainLabels);
 //   saveDataFile(this->TestFileName,dataTestDescriptor,this->TestLabels);
    std::cout<<"\n saved "<< this->TestFileName<< "and"<< this->TrainFileName<<std::endl;

}

void BagOfSIFT::saveDataFile(const std::string &file_name, cv::Mat &dataDescriptor, cv::Mat &label) {

    cv::FileStorage fs(file_name,cv::FileStorage::WRITE);
    if(!fs.isOpened())
        std::cout<<"file could not be opened\n";
    fs<< "descriptors" << dataDescriptor;
    fs<< "labels" << label;
    //release the file storage
    fs.release();

}

void BagOfSIFT::loadDataFile(const std::string &file_name, cv::Mat &dataDescriptor, cv::Mat &label) {


    cv::FileStorage fs(file_name,cv::FileStorage::READ);
    if(!fs.isOpened())
        std::cout<<"file could not be opened\n";
    fs["descriptors"] >> dataDescriptor;
    fs["labels"] >> label;
    fs.release();


}

void BagOfSIFT::BuildBOF() {





    //set keypointsize
    this->keypointsize = 16;
    this->FileName = "dictionary_" + this->DictionarySizeString +"_"+ this->StepSizeString + ".yml";
    //The SIFT feature extractor and descriptor
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    //To store all the descriptors that are extracted from all the images.
    cv::Mat Img;
    //temporal storage of the SIFT descriptor of the current image
    cv::Mat1f descriptor;
    //To store the keypoints that will be extracted by SIFT
    std::vector<cv::KeyPoint> KeyPoints;
    cv::KeyPoint KeyPoint;

    //Extract SIFT features
    for (int i = 0; i <this->TrainImages.size(); ++i){
        float percentage = (float)(i+1)/(float)this->TrainImages.size()*100;
        Img = this->TrainImages[i];

        //sanity check
        if(!Img.data)
        {
            std::cerr << "Problem loading image!!!" << std::endl;
            continue;
        }

        //Detect SIFT features and compute descriptors

        for (int j = StepSize; j <Img.rows ; j+=StepSize) {



            for (int k = StepSize ; k < Img.cols ; k+=StepSize) {
                KeyPoint = cv::KeyPoint(cv::Point2f(j,k),keypointsize);
                KeyPoints.push_back(KeyPoint);
            }

        }

        //f2d->detect(Img,KeyPoints);

        std::cout<<"\r Computing "<<percentage<<"% done" ;
        f2d->compute(Img,KeyPoints,descriptor);
        KeyPoints.clear();
        this->featuresUnclustered.push_back(descriptor);

    }

    std::cout<<"\r BOW Construction is starting..."<<std::endl;

    //Construct BOWKMeansTrainer
    //the number of bags
    cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    tc.epsilon=FLT_EPSILON;
    //retries number
    int retries=1;
    //necessary flags
    int flags = cv::KMEANS_PP_CENTERS;
    //Create BOW trainer
    cv::BOWKMeansTrainer bowTrainer(this->dictionarysize,tc,retries,flags);
    //cluster feature vectors
    this->dictionary = bowTrainer.cluster(featuresUnclustered);


    //Store the vocabulary
    cv::FileStorage fs(this->FileName,cv::FileStorage::WRITE);
    fs << "vocabulary"<< this->dictionary;
    fs.release();
    std::cout<<" saved "<< FileName <<std::endl;

}


BagOfSIFT::~BagOfSIFT() {

}
