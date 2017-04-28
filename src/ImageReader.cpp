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
         //   std::cout<<train_row[i]<<std::endl;
            temp2 = cv::imread(train_row[i],CV_LOAD_IMAGE_GRAYSCALE);
            //temp2.convertTo(temp2,CV_32FC1);
            this->Train_Images.push_back(temp2);
            this->Train_Labels.push_back(k);
        }

        train_row.clear();

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
             char filename[128];
            Mat frame;
            int frame_count;
            bool should_stop = false;
            double fIdx;
            bool success = cap.read(frame); 
            if ( ! success ) {
                cout << "Cannot read frame " << endl;
                break;
                }
          //  temp2 = cv::imread(frame,CV_LOAD_IMAGE_GRAYSCALE);
            this->Test_Images.push_back(frame) ;   //this - pointer la obiectul curent
             sprintf(filename, "../predictie/frame_%06d.jpg", frame_count);
                    cv::imwrite(filename, frame);
                    frame_count++;

           




            this->Test_Labels.push_back(0);

            //cap >> frame; //get a new frame from camera 
            //if (frame.empty()) continue;

            imshow("tree", frame);
            //if ( waitKey (0) == 27 ) break;
            waitKey(1);
            index++;


    // for (int i = 0; i < numOfTestInputs; i++) {
    //          std::stringstream ss9;
    //          ss9 << "../predictie/robo_";
    //         ss9 << i;
    //         ss9 << ".png";
    //         std::string name = ss9.str();
    //         std::cout <<name<<std::endl;
    //         imwrite(name, IMREAD_COLOR);} 

        }




        

    std::cout<<"reading done..."<<std::endl;


}

ImageReader::~ImageReader() {

}
