#ifndef PRJ3_008284_010097_BAGOFSIFT_H
#define PRJ3_008284_010097_BAGOFSIFT_H

#include<iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "ImageReader.h"

#define NUM_OF_QUADRANTS  16  // must be powers of two 

class BagOfSIFT {

public:
    BagOfSIFT(ImageReader *DataSet);
    cv::Mat dictionary;
    cv::Mat1f dataTestDescriptor;
    cv::Mat1f dataTrainDescriptor;
    cv::Mat TestLabels;
    cv::Mat TrainLabels;

    ~BagOfSIFT();
    const std::string strTrain = {"train"};
    const std::string strTest = {"test"};
    void saveDataFile(const std::string &file_name, cv::Mat& dataDescriptor, cv::Mat& label);
    void loadDataFile(const std::string &file_name, cv::Mat& dataDescriptor, cv::Mat& label);


    // NM
    std::vector<cv::Mat1f> dataTestQuadrantDescriptor;
    std::vector<cv::Mat1f> dataTrainQuadrantDescriptor;

private:

    cv::Mat1f featuresUnclustered;
    //generate visiual vocabulary
    void BuildBOF();
    int dictionarysize;
    int StepSize;
    //extract BOF of all test images
    void Extract_BOF_features();
    float keypointsize;
    std::string Read;
    std::string Write;
    std::string StepSizeString;
    std::string DictionarySizeString;
    std::string FileName;
    std::string TestFileName;
    std::string TrainFileName;

    std::vector<cv::Mat> TestImages;
    std::vector<cv::Mat> TrainImages;

};


#endif //PRJ3_008284_010097_BAGOFSIFT_H
