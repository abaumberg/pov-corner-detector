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
#define THRESHOLD 10000.0f
#define sigmaD 3.0
#define sigmaI 3.0

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

    Mat Gx(inputImage.rows, inputImage.cols, CV_32F);
    Mat Gy(inputImage.rows, inputImage.cols, CV_32F);
    Mat cornerRImage(inputImage.rows, inputImage.cols, CV_32F);

    Mat sobelImage(inputImage.rows, inputImage.cols, inputImage.depth());
    Mat cornerImage(inputImage.rows, inputImage.cols, inputImage.depth());

    Mat gaussianImage(inputImage.rows, inputImage.cols, inputImage.depth());


    ////////////////////////////////////////////////////////////////////////////
    // SOBEL
    ////////////////////////////////////////////////////////////////////////////
    GaussianBlur(inputImage, inputImage, Size(3,3), sigmaD);

    Sobel(inputImage, Ix, inputImage.depth(), 0, 1, 3);
    Sobel(inputImage, Iy, inputImage.depth(), 1, 0, 3);


    for (int i=0; i<inputImage.rows; i++) {
        for (int j=0; j<inputImage.cols; j++) {
            sobelImage.at<uchar>(i,j) =
                round(
                    sqrt(
                        Ix.at<uchar>(i,j) * Ix.at<uchar>(i,j) +
                        Iy.at<uchar>(i,j) * Iy.at<uchar>(i,j)
                    )
                );
        }
    }


    imshow("Sobel", sobelImage);
    waitKey();


    ////////////////////////////////////////////////////////////////////////////
    // HARRIS CORNER DETECTOR
    ////////////////////////////////////////////////////////////////////////////

    GaussianBlur(Ix, Gx, Size(3,3), sigmaI);
    GaussianBlur(Iy, Gy, Size(3,3), sigmaI);


    int counter = 0;
    Mat movementMatrix(2,2, CV_32F);

    for (int i=0; i<Gx.rows; i++) {
        for (int j=0; j<Gx.cols; j++) {
            /*float x = Gx.at<float>(i,j);
            float y  = Gy.at<float>(i,j);
            float xx = x * x;
            float yy = y * y;
            float xy = x * y;

            movementMatrix.at<float>(0,0) = xx;
            movementMatrix.at<float>(0,1) = xy;
            movementMatrix.at<float>(1,0) = xy;
            movementMatrix.at<float>(1,1) = yy;

            movementMatrix *= sigmaD * sigmaD;

            float det = determinant(movementMatrix);
            Vec<float,4> tmp = trace(movementMatrix);
            float tr = tmp[0];


            float R = fabs(det - K * tr * tr);*/

            cornerRImage.at<float>(i,j) = Gx.at<float>(i,j);
        }
    }

    imshow("Gx", Gx);
    imshow("R", cornerRImage);
    waitKey();

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
                            //condition = false;
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

    imshow("Corners", cornerImage);
    waitKey();

    return EXIT_SUCCESS;
}
