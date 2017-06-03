/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek   S008284 Department of Electrical and Electronics
        Baris Ozcan S010097 Department of Computer Science

*/

#ifndef PRJ3_008284_010097_IMAGEREADER_H
#define PRJ3_008284_010097_IMAGEREADER_H

#include <vector>
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define NUM_OF_CLASS  2
#define NUM_OF_TYPES 2

class coordinates{ 
public:
     int t, l, h, w;

     // bool isInside(int x, int y)
     // {
     //    return (x > l && x < l+w && y > t && y < t+h);
     // }

     bool isInside(int dl, int dt, int dw, int dh)
     {
        int dCenterX = dl + dw/2;
        int dCenterY = dt + dh/2;
        int centerX = l + w/2;
        int centerY = t + h/2;

        return (((dCenterX >= l && dCenterX <= l+w) && (dCenterY >= t && dCenterY <= t+h)) || ((centerX >= dl && centerX <= dl+dw) && (centerY >= dt && centerY <= dt+dh)));
     }
};

// ImageReader, reads images from specified folders and creates a vector of cv::Mat on grayscale with their labels stored at Train/Test_Labels.

class ImageReader {

public:
    ImageReader();
     std::vector<coordinates> vr;
    std::vector<cv::Mat> Train_Images;
    std::vector<cv::Mat> Test_Images;
    cv::Mat Test_Labels;
    cv::Mat Train_Labels;
      std::string name;
    ~ImageReader();
private:


    void readFilenamesBoost(std::vector<std::string> &filenames, const std::string &folder);
    void ExtractFiles();

    std::string types_class_folder[NUM_OF_TYPES] = {"test/", "train/"};
                                               
    std::string class_train_folders[NUM_OF_CLASS] = {"train/pos/","train/neg/"};
                                                     
    std::string class_test_folders[NUM_OF_CLASS] = {"test/postest/","test/negtest/"};

    std::string folder = "../../project3-data/";
};



#endif //PRJ3_008284_010097_IMAGEREADER_H
