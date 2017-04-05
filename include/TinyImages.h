/*
  Muhammet Pakyürek	S008284 Department of Electrical and Electronics
  Baris Ozcan	S010097	Department of Computer Science
*/



#ifndef PRJ3_008284_010097_TINYIMAGES_H
#define PRJ3_008284_010097_TINYIMAGES_H

#include<iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ImageReader.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

class TinyImages {

public:
    TinyImages(ImageReader *DataSet);
    void TinyImageTest();
    cv::Ptr<cv::ml::KNearest> kn;
    std::vector<cv::Mat> normalizedTrainImages;
    std::vector<cv::Mat> normalizedTestImages;
    ~TinyImages();

private:


    void normalizeData(); // Normalizes AND vectorizes the image!
    cv::Mat testDatavector;
    cv::Mat trainDatavector;
    std::vector<cv::Mat> TestImages;
    std::vector<cv::Mat> TrainImages;
    cv::Mat TestLabels;
    cv::Mat TrainLabels;

};


#endif //PRJ3_008284_010097_TINYIMAGES_H
