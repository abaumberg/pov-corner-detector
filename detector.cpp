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
#define THRESHOLD 127
#define S 0.7
#define epsilon 1.4


using namespace std;
using namespace cv;

void detectCorners(Mat &src, Mat &dst, int blockSize, int kernelSize) {

    Mat Ix, Iy, Ixx;

    Sobel(src, Ix, CV_32F, 1, 0, kernelSize);
    Sobel(src, Iy, CV_32F, 0, 1, kernelSize);

    Size size = src.size();
    Mat covariance( size, CV_32FC3 );

    for(int i=0; i<covariance.rows; i++ ) {
        for(int j = 0; j<covariance.cols; j++ ) {

            float x = Ix.at<float>(i,j);
            float y = Iy.at<float>(i,j);

            Vec3f v;
            v[0] = x*x;
            v[1] = x*y;
            v[2] = y*y;

            covariance.at<Vec3f>(i,j) = v;
        }
    }

    boxFilter(covariance, covariance, covariance.depth(), Size(blockSize, blockSize));

    if(covariance.isContinuous() && covariance.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }

    for(int i=0 ; i < covariance.rows; i++ ) {
        for(int j = 0; j < covariance.cols; j++ ){

            Vec3f v = covariance.at<Vec3f>(i,j);
            v *= kernelSize * kernelSize;
            float a = v[0];
            float b = v[1];
            float c = v[2];
            dst.at<float>(i,j) = (float)(a*c - b*b - K*(a + c)*(a + c));
        }
    }
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

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( grayImage.size(), CV_32FC1 );

    /// Detector parameters

    int apertureSize = 5;

    for (float b=3; b<10; b++) {


        //cout << "Scale: " << s << endl;
        detectCorners(grayImage, dst, b, apertureSize);
        //cornerHarris( grayImage, dst, b, apertureSize, k, BORDER_DEFAULT );

        /// Normalizing
        normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        convertScaleAbs( dst_norm, dst_norm_scaled );
        int counter = 0;
        Mat tmp = inputImage.clone();
        for( int j = 0; j < dst_norm.rows ; j++ ) {
            for( int i = 0; i < dst_norm.cols; i++ ) {
                if( (int) dst_norm.at<float>(j,i) > THRESHOLD) {
                    circle( tmp, Point( i, j ), 5,  Scalar(255, 0, 0), 2, 8, 0 );
                    counter++;
                }
            }
        }
        cout << "Corners: " << counter << endl;
        /// Showing the result
        imshow("Corners", tmp );
        waitKey();

    }

    return EXIT_SUCCESS;
}
