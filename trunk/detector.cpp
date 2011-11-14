#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    string inputImageName;

    // zpracovani parametru prikazove radky
    for( int i = 1; i < argc; i++){
        if( string(argv[ i]) == "-ii" && i + 1 < argc){
            inputImageName = argv[ ++i];
        } else if( string(argv[ i]) == "-h"){
            cout << "Use: " << argv[0] << "  -ii inputImageName" << endl;
            return EXIT_SUCCESS;
        } else {
            cerr << "Error: Unrecognized command line parameter \"" << argv[ i] << "\" use -h to get more information." << endl;
        }
    }


    if( inputImageName.empty()){
        cerr << "Error: Some mandatory command line options were not specified." << endl;
        return EXIT_FAILURE;
    }

    Mat inputImage = cvLoadImageM(inputImageName.c_str());

    cvtColor(inputImage, inputImage, CV_RGB2GRAY);
    imshow("Img", inputImage);
    waitKey();

    Mat verticalGradient(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat horizontalGradient(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat outputImage(inputImage.rows, inputImage.cols, inputImage.depth());

    Sobel(inputImage, verticalGradient, inputImage.depth(), 1, 0, 3);
    Sobel(inputImage, horizontalGradient, inputImage.depth(), 0, 1, 3);

    for (int i=0; i<inputImage.rows; i++) {
        for (int j=0; j<inputImage.cols; j++) {
            outputImage.at<char>(i,j) =
                round(
                    fabs(verticalGradient.at<char>(i,j)) +
                    fabs(horizontalGradient.at<char>(i, j))
                );

        }
    }

    imshow("Img", outputImage);
    waitKey();

    return EXIT_SUCCESS;
}
