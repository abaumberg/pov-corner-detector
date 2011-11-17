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
#define THRESHOLD 1e7
#define S 0.7
#define epsilon 1.4


using namespace std;
using namespace cv;

float kernelW(int x, int y, int sigma) {
    int sigmaSqr = sigma * sigma;
    int xSqr = x * x;
    int ySqr = y * y;
    return pow(M_E, -((xSqr + ySqr) / (2 * sigmaSqr))) / (2 * M_PI * sigmaSqr);
}

void HarrisDetector(Mat src, Mat &dst, float sigmaI, float sigmaD) {

    Mat Ix(src.rows, src.cols, src.depth());
    Mat Iy(src.rows, src.cols, src.depth());

    Mat Gx(src.rows, src.cols, src.depth());
    Mat Gy(src.rows, src.cols, src.depth());
    Mat Lx(src.rows, src.cols, src.depth());
    Mat Ly(src.rows, src.cols, src.depth());

    Mat laplacianImage(src.rows, src.cols, src.depth());
    Mat cornerRImage(src.rows, src.cols, CV_16U);

    Mat sobelImage(src.rows, src.cols, src.depth());

    Mat gaussianImage(src.rows, src.cols, src.depth());

    ////////////////////////////////////////////////////////////////////////////
    // SOBEL
    ////////////////////////////////////////////////////////////////////////////
    GaussianBlur(src, gaussianImage, Size(3,3), sigmaD);


    Sobel(gaussianImage, Ix, gaussianImage.depth(), 0, 1, 3);
    Sobel(gaussianImage, Iy, gaussianImage.depth(), 1, 0, 3);

    /*for (int i=0; i<Ix.rows; i++) {
        for (int j=0; j<Ix.cols; j++) {
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
    Laplacian(Ix, Lx, gaussianImage.depth(), 3);
    Laplacian(Iy, Ly, gaussianImage.depth(), 3);


    int counter = 0;
    Mat movementMatrix(2,2, CV_32F);

    uint* cornerR = new uint[Gx.rows*Gx.cols];
    float* LoG = new float[Gx.rows*Gx.cols];

    for (int i=0; i<Gx.rows; i++) {
        for (int j=0; j<Gx.cols; j++) {
            uint x = Gx.at<uchar>(i,j);
            uint y = Gy.at<uchar>(i,j);

            float xx = x * x;
            float yy = y * y;
            float xy = x * y;

            xx *= sigmaD * sigmaD;
            yy *= sigmaD * sigmaD;
            xy *= sigmaD * sigmaD;

            float det = xx*yy - xy*xy;
            float tr = xx + yy;
            uint R = round(fabs(det - K * tr * tr));

            cornerR[i+j*Gx.rows] = R;

            LoG[i+j*Gx.rows] =
                sigmaI * sigmaI * abs(Lx.at<char>(i,j) + Ly.at<char>(i,j));
            //laplacianImage.at<uchar>(i,j) =
                //round();
        }
    }

    Mat cornerImage(src.rows, src.cols, src.depth());
    cornerImage.setTo(0);

    // 8-point neighbourhood filtering
    for (int i=1; i<cornerRImage.rows-1; i++) {
        for (int j=1; j<cornerRImage.cols-1; j++) {

            uint center = cornerR[i+j*cornerRImage.rows];
            bool condition = true;

            if (center > THRESHOLD) {
                for (int m=-1; m<=1; m++) {
                    for (int n=-1; n<=1; n++) {
                        if (n == 0 && m == 0) continue;

                        uint neighbour = cornerR[i+m+(j+n)*cornerRImage.rows];

                        if (neighbour >= center) {
                            condition = false;
                        }
                    }
                }

                if (condition) {
                    counter++;
                    cornerImage.at<uchar>(i,j) = 255;
                }

            }
        }
    }

    delete [] cornerR;
    delete [] LoG;

    cout << counter << endl;
    //cout << cornerImage.depth() << endl;
    //cout << inputImage.depth() << endl;

    imshow("Corners", cornerImage);


    waitKey();
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
    Mat grayImage(inputImage.rows, inputImage.cols, 0);

    cvtColor(inputImage, grayImage, CV_RGB2GRAY);
    imshow("Img", grayImage);
    waitKey();


    float sigma0 = 1.0f;
    float sigmaI = sigma0;
    float sigmaD = 0;
    for (int i=0; i<10; i++) {
        sigmaD = sigmaI * S;
        HarrisDetector(grayImage, inputImage, sigmaI, sigmaD);
        sigmaI *= epsilon;
    }

    return EXIT_SUCCESS;
}
