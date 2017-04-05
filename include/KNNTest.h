/*
Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science
*/

#ifndef PRJ3_008284_010097_KNNTEST_H
#define PRJ3_008284_010097_KNNTEST_H


#include "BagOfSIFT.h"
#include<iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "ImageReader.h"

class KNNTest {

public:

    KNNTest(BagOfSIFT *BagOfSIFT);
    ~KNNTest();

private:

    cv::Ptr<cv::ml::KNearest> kn;
    cv::Mat1f dataTrainDescriptor;
    cv::Mat1f dataTestDescriptor;
    cv::Mat trainLabels;
    cv::Mat testLabels;




};


#endif //PRJ3_008284_010097_KNNTEST_H
