/*
  Muhammet Pakyürek	S008284 Department of Electrical and Electronics
  Baris Ozcan	S010097	Department of Computer Science
*/

#include "opencv2/opencv.hpp"
#include "iostream"
#include <string>
#include <stdlib.h>

#include "ImageReader.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

ImageReader::ImageReader() {

    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------Reading Images----------"<<std::endl;
    std::cout<<"     "<<std::endl;


    ImageReader::ExtractFiles();

}

void ImageReader::readFilenamesBoost(std::vector<std::string> &filenames, const std::string &folder)
{
    boost::filesystem::path directory(folder);
    boost::filesystem::directory_iterator itr(directory), end_itr;



    for(;itr != end_itr; ++itr)
    {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (is_regular_file(itr->path()))
        {
            // assign current file name to current_file and echo it out to the console.
            //            string full_file_name = itr->path().string(); // returns full path
            std::string filename = itr->path().string(); // returns just filename
            filenames.push_back(filename);
        }
    }
}

void ImageReader::ExtractFiles() {

    std::vector<std::string> test_row;
    std::vector<std::string> train_row;
    std::cout<<"reading images..."<<std::endl;
    cv::Mat temp1; // container for temporary images.
    cv::Mat temp2;

    //for each class
    for(int k = 0; k<NUM_OF_CLASS; k++)
    {

        std::string train_appnd_folder(folder  + class_train_folders[k]);
        //get full path of all files in appnd_folder
        readFilenamesBoost(train_row, train_appnd_folder);

        for (int i = 0; i < train_row.size() ; ++i) {
            std::cout<<train_row[i]<<std::endl;
            temp2 = cv::imread(train_row[i],CV_LOAD_IMAGE_GRAYSCALE);
            //temp2.convertTo(temp2,CV_32FC1);
            this->Train_Images.push_back(temp2);
            this->Train_Labels.push_back(k);
        }

        train_row.clear();
        //for example appnd_folder = "/home/magneficent/cv_projects/Project3/project3_data/train/bedroom"
        // std::string test_appnd_folder(folder + class_test_folders[k]); //appnd_folder.append(folder,class_folders[k]);
        // readFilenamesBoost(test_row, test_appnd_folder);
        // for (int i = 0; i < test_row.size() ; ++i) {
        //     temp1 = cv::imread(test_row[i],CV_LOAD_IMAGE_GRAYSCALE);
        //     this->Test_Images.push_back(temp1) ;
        //     this->Test_Labels.push_back(k);

        // }
        // test_row.clear();

    }

    
        VideoCapture cap( "robot1.avi" ); // open the default camera
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); 
        if( ! cap.isOpened () )  // check if we succeeded
        return ;

        //readFilenamesBoost(test_row, cap);

        namedWindow ( "robot1" , 1 );
        double frnb ( cap.get ( CV_CAP_PROP_FRAME_COUNT ) );
        std::cout << "frame count = " << frnb << endl;

        int index=0;

        for(;;) {
            Mat frame;
            double fIdx;
            bool success = cap.read(frame); 
            if ( ! success ) {
                cout << "Cannot read frame " << endl;
                break;
                }
          //  temp2 = cv::imread(frame,CV_LOAD_IMAGE_GRAYSCALE);
            this->Test_Images.push_back(frame) ;   //this - pointer la obiectul curent
            this->Test_Labels.push_back(0);

            //cap >> frame; //get a new frame from camera 
            //if (frame.empty()) continue;

            imshow("tree", frame);
            //if ( waitKey (0) == 27 ) break;
            waitKey(1);
            index++;
        }
    

    // VideoCapture capture;
    // Mat frame;

    // capture.open( -1 );
    // if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return ; }
    // while (true) {
    //   capture.read(frame);
    //   imshow("tree", frame); 
    
    // {

    //     if( frame.empty() )
    //     {
    //         cout<<(" --(!) No captured frame -- Break!")<<endl;
    //        // break;
    //     }
    //     //--Apply the classifier to the frame
    //   //  detectAndDisplay( frame );
    //     int c = waitKey(10);
    //     if( (char)c == 27 ) { break; } // escape
    // }
    //     }
    // return ;


    std::cout<<"reading done..."<<std::endl;


}

ImageReader::~ImageReader() {

}
