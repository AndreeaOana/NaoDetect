/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek   S008284 Department of Electrical and Electronics
        Baris Ozcan S010097 Department of Computer Science

*/

#include "opencv2/opencv.hpp"
#include "iostream"
#include <string>
#include <stdlib.h>
#include <fstream>
#include <list>


#include "ImageReader.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

int top, left, height, width;


void rotate_90n(cv::Mat const &src, cv::Mat &dst, int angle)
{        
     CV_Assert(angle % 90 == 0 && angle <= 360 && angle >= -360);
     if(angle == 270 || angle == -90){
        // Rotate clockwise 270 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 0);
    }else if(angle == 180 || angle == -180){
        // Rotate clockwise 180 degrees
        cv::flip(src, dst, -1);
    }else if(angle == 90 || angle == -270){
        // Rotate clockwise 90 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 1);
    }else if(angle == 360 || angle == 0 || angle == -360){
        if(src.data != dst.data){
            src.copyTo(dst);
        }
    }
}

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

    string filename = "testing.xml";

    string line;
    size_t pos;

    coordinates coord;

    std::string t0 = "top='";
    std::string t1 = "' left='";
    std::string t2 = "' width='";
    std::string t3 = "' height='";
    std::string t4 = "'/>";
    bool lookForImage = true;

    //lista

  //  std::list<coordonates> coord;

    //coord.push_back();

// BIBLIO:
// -> ce e in conlcluzie de trecut la sinteza

// 3.  Contribution
//     3.1 Research question
//     3.2 Proposition
//     3.3 System description
//     3.4 Running example
// 4. Evaluation
// 5. Synthesis and future work

    ifstream myfile (filename);
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            pos=line.find("<image"); // search
            if (pos!=string::npos)
            {
                if (!lookForImage)
                {
                    std::cerr << "Image without box !!" << std::endl;
                }
                lookForImage = false;
            }
            //cout << line << '\n';
            pos=line.find("box"); // search
            if(pos!=string::npos) // string::npos is returned if string is not found
            {
                size_t pos0 = line.find(t0);
                size_t pos1 = line.find(t1);
                size_t pos2 = line.find(t2);
                size_t pos3 = line.find(t3);
                size_t pos4 = line.find(t4);

                int top = std::stoi(line.substr(pos0+t0.size(),pos1));
                int left = std::stoi(line.substr(pos1+t1.size(),pos2));
                int width = std::stoi(line.substr(pos2+t2.size(),pos3));
                int height = std::stoi(line.substr(pos3+t3.size(),pos4));

                std::cerr << top << " " << left << " " << width << " " << height << std::endl;

                coord.t = top;
                coord.l = left;
                coord.w = width;
                coord.h = height;

                vr.push_back(coord);

                

                lookForImage = true;
            }
        }
        myfile.close();
    }

  else cout << "Unable to open file"; 
        
    
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
        {
            std::cout << "video not opened" << endl;    
            return ;
        }


        double frnb ( cap.get ( CV_CAP_PROP_FRAME_COUNT ) );
        std::cout << "frame count = " << frnb << endl;

        int index=0;

        for(;;) {
            char filename[128];
            Mat framecolor;
            int frame_count;
            bool should_stop = false;
            double fIdx;
            bool success = cap.read(framecolor); 
            if ( ! success ) {
                cout << "Cannot read frame " << endl;
                break;
                }
          //  temp2 = cv::imread(frame,CV_LOAD_IMAGE_GRAYSCALE);
            Mat frame;
            cvtColor(framecolor, frame, CV_RGB2GRAY); 
            imshow("RobotVideo", frame); 
            
            //rotate_90n(frame,frame,180);

            this->Test_Images.push_back(frame) ;   //this - pointer la obiectul curent

            sprintf(filename, "../predictie/frame_%06d.jpg", frame_count);
            cv::imwrite(filename, frame);
            frame_count++;


            this->Test_Labels.push_back(0);

           

         //   imshow("tree", frame);
            waitKey(1);
            index++;

        }

      //  std::cerr << "test labels size" << this->Test_Labels.size() << std::endl;
        

    std::cout<<"reading done..."<<std::endl;


}

ImageReader::~ImageReader() {

}
