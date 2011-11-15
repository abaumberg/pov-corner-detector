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
#define K 0.04
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
    Mat grayImage(inputImage.rows, inputImage.cols, 0);

    cvtColor(inputImage, grayImage, CV_RGB2GRAY);
    imshow("Img", grayImage);
    waitKey();

    Mat Ix(grayImage.rows, grayImage.cols, grayImage.depth());
    Mat Iy(grayImage.rows, grayImage.cols, grayImage.depth());

    Mat Gx(grayImage.rows, grayImage.cols, grayImage.depth());
    Mat Gy(grayImage.rows, grayImage.cols, grayImage.depth());
    Mat cornerRImage(grayImage.rows, grayImage.cols, CV_16U);

    Mat sobelImage(grayImage.rows, grayImage.cols, grayImage.depth());
    //Mat cornerImage(grayImage.rows, grayImage.cols, grayImage.depth());

    Mat gaussianImage(grayImage.rows, grayImage.cols, grayImage.depth());


    ////////////////////////////////////////////////////////////////////////////
    // SOBEL
    ////////////////////////////////////////////////////////////////////////////
    GaussianBlur(grayImage, gaussianImage, Size(3,3), sigmaD);


    Sobel(gaussianImage, Ix, gaussianImage.depth(), 0, 1, 3);
    Sobel(gaussianImage, Iy, gaussianImage.depth(), 1, 0, 3);


    /*for (int i=0; i<inputImage.rows; i++) {
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
    */

    ////////////////////////////////////////////////////////////////////////////
    // HARRIS CORNER DETECTOR
    ////////////////////////////////////////////////////////////////////////////

    GaussianBlur(Ix, Gx, Size(3,3), sigmaI);
    GaussianBlur(Iy, Gy, Size(3,3), sigmaI);


    int counter = 0;
    Mat movementMatrix(2,2, CV_32F);

    uint* c = new uint[Gx.rows*Gx.cols];

    for (int i=0; i<Gx.rows; i++) {
        for (int j=0; j<Gx.cols; j++) {
            uint x = Gx.at<uchar>(i,j);
            uint y = Gy.at<uchar>(i,j);
            uint xx = x * x;
            uint yy = y * y;
            uint xy = x * y;

            movementMatrix.at<float>(0,0) = xx;
            movementMatrix.at<float>(0,1) = xy;
            movementMatrix.at<float>(1,0) = xy;
            movementMatrix.at<float>(1,1) = yy;

            float det = determinant(movementMatrix);
            Vec<float,4> tmp = trace(movementMatrix);
            float tr = tmp[0];


            uint R = fabs(det - K * tr * tr);

            c[i+j*cornerRImage.rows] = R;
        }
    }

    imshow("Gx", Gx);
    imshow("R", cornerRImage);
    waitKey();

    Mat cornerImage(grayImage.rows, grayImage.cols, grayImage.depth());

    cout << cornerRImage.cols << " " << cornerRImage.rows << endl;
    cout << cornerImage.cols << " " << cornerImage.rows << endl;

    // 8-point neighbourhood filtering
    for (int i=1; i<cornerRImage.rows-1; i++) {
        for (int j=1; j<cornerRImage.cols-1; j++) {

            uint center = c[i+j*cornerRImage.rows];
            bool condition = true;

            if (center > THRESHOLD) {
                for (int m=-1; m<=1; m++) {
                    for (int n=-1; n<=1; n++) {
                        if (n == 0 && m == 0) continue;

                        uint neighbour = c[i+m+(j+n)*cornerRImage.rows];

                        if (neighbour >= center) {
                            condition = false;
                        }
                    }
                }

                if (condition) {
                    counter++;
                    cornerImage.at<uchar>(i,j) = 255;//grayImage.at<uchar>(i,j);
                }

            }

        }
    }



    cout << counter << endl;
    //cout << cornerImage.depth() << endl;
    //cout << inputImage.depth() << endl;

    imshow("Corners", cornerImage);
    waitKey();

    return EXIT_SUCCESS;
}
