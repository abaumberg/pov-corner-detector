#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>

/**
 *
 * http://en.wikipedia.org/wiki/Harris_affine_region_detector
 *
 */

// Empirical constant 0.04-0.06
#define K 0.06

using namespace std;
using namespace cv;

float kernelW(int x, int y, int sigma) {
    int sigmaSqr = sigma * sigma;
    int xSqr = x * x;
    int ySqr = y * y;
    return pow(M_E, -((xSqr + ySqr) / (2 * sigmaSqr))) / (2 * M_PI * sigmaSqr);
}

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
            cerr << "Error: Unrecognized command line parameter \""
                 << argv[ i] << "\" use -h to get more information." << endl;
        }
    }


    if( inputImageName.empty()){
        cerr << "Error: Some mandatory command line options were not specified."
             << endl;
        return EXIT_FAILURE;
    }

    Mat inputImage = cvLoadImageM(inputImageName.c_str());

    cvtColor(inputImage, inputImage, CV_RGB2GRAY);
    imshow("Img", inputImage);
    waitKey();

    Mat Ix(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat Iy(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat outputImage(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat cornerImage(inputImage.rows, inputImage.cols, inputImage.depth());


    ////////////////////////////////////////////////////////////////////////////
    // SOBEL
    ////////////////////////////////////////////////////////////////////////////
    Sobel(inputImage, Iy, inputImage.depth(), 1, 0, 3);
    Sobel(inputImage, Ix, inputImage.depth(), 0, 1, 3);

    for (int i=0; i<inputImage.rows; i++) {
        for (int j=0; j<inputImage.cols; j++) {
            outputImage.at<uchar>(i,j) =
                round(fabs(Ix.at<uchar>(i,j)) + fabs(Iy.at<uchar>(i, j)));
        }
    }


    imshow("Img", outputImage);
    waitKey();

    ////////////////////////////////////////////////////////////////////////////
    // HARRIS CORNER DETECTOR
    ////////////////////////////////////////////////////////////////////////////
    int counter = 0;
    Mat movementMatrix(4,4, CV_32F);
    for (int i=0; i<inputImage.rows; i++) {
        for (int j=0; j<inputImage.cols; j++) {
            uint x = Ix.at<uchar>(j,i);
            uint y = Ix.at<uchar>(j,i);
            uint xx = x * x;
            uint yy = y * y;
            uint xy = x * y;

            float kernelSum = 0.0f;

            for (int m=0; m<5; m++) {
                for (int n=0; n<5; n++) {
                    kernelSum = kernelW(m,n, 5.0);
                }
            }

            movementMatrix.at<float>(0,0) = xx * kernelSum;
            movementMatrix.at<float>(0,1) = xy * kernelSum;
            movementMatrix.at<float>(1,0) = xy * kernelSum;
            movementMatrix.at<float>(1,1) = yy * kernelSum;

            float det = determinant(movementMatrix);
            Vec<float,4> tmp = trace(movementMatrix);
            float tr = tmp[0];


            float R = fabs(det - K * tr * tr);

            if (R > 1000.0f) {
                //cout << R << endl;
                counter++;
                cornerImage.at<uchar>(j,i) = inputImage.at<uchar>(j,i);

            }
        }
    }

    cout << counter << endl;

    imshow("Img1", inputImage);
    imshow("Img2", cornerImage);
    waitKey();

    return EXIT_SUCCESS;
}
