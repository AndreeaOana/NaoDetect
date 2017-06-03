/*
author Andreea-Oana Petac, MRI Project, ENIB

used parts of code from Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science

*/

#include <iostream>
#include "vector"
#include "opencv2/core.hpp"
#include "ImageReader.h"
#include "TinyImages.h"
#include "BagOfSIFT.h"
#include "KNNTest.h"
#include "SVMAnalysis.h"


using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    std::cout<<std::sqrt(16);
    


    ImageReader ImageRead;

//    TinyImages TinyImages(&ImageRead);
    BagOfSIFT BagOfSIFT(&ImageRead);
//    KNNTest KNNTest(&BagOfSIFT);
    SVMAnalysis SVMAnalysis(&BagOfSIFT, &ImageRead);




    return 0;
}
