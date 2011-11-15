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
#define THRESHOLD 1000.0f

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
                 << argv[ i] << "\" use -h to get mo re information." << endl;
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
    Mat cornerRImage(inputImage.rows, inputImage.cols, CV_32F);


    ////////////////////////////////////////////////////////////////////////////
    // SOBEL
    ////////////////////////////////////////////////////////////////////////////
    Sobel(inputImage, Iy, inputImage.depth(), 1, 0, 3);
    Sobel(inputImage, Ix, inputImage.depth(), 0, 1, 3);

    for (int i=0; i<inputImage.rows; i++) {
        for (int j=0; j<inputImage.cols; j++) {
            outputImage.at<uchar>(i,j) =
                round(
                    sqrt(
                        Ix.at<uchar>(i,j) * Ix.at<uchar>(i,j) +
                        Iy.at<uchar>(i,j) * Iy.at<uchar>(i,j)
                    )
                );
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
            uint x = Ix.at<uchar>(i,j);
            uint y = Ix.at<uchar>(i,j);
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

            cornerRImage.at<float>(i,j) = R;
        }
    }

    // 8-point neighbourhood filtering
    for (int i=1; i<cornerImage.rows-1; i++) {
        for (int j=1; j<cornerImage.cols-1; j++) {
            float center = cornerRImage.at<float>(i,j);
            bool condition = true;

            if (center > THRESHOLD) {
                for (int m=-1; m<=1; m++) {
                    for (int n=-1; n<=1; n++) {
                        if (n == 0 && m == 0) continue;

                        float neighbour = cornerRImage.at<float>(i+m,j+n);

                        if (neighbour >= center) {
                            condition = false;
                        }
                    }
                }

                if (condition) {
                    counter++;
                    cornerImage.at<uchar>(i,j) = inputImage.at<uchar>(i,j);
                }

            }
        }
    }

    cout << counter << endl;

    imshow("Img1", inputImage);
    imshow("Img2", cornerImage);
    waitKey();

    return EXIT_SUCCESS;
}
