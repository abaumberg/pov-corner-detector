#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include "math.h"

/**
 *
 * http://en.wikipedia.org/wiki/Harris_affine_region_detector
 *
 */

// Empirical constant 0.04-0.06
#define K 0.04
#define THRESHOLD 127
#define THRESHOLD2 10000.0
#define S 0.7
#define epsilon 1.4

//double const Pi=4*atan(1);


using namespace std;
using namespace cv;

float gaussianDerivate(int x, int y, float sigma) {
	return (-x * pow(M_E, (-(x*x+y*y)/(2*sigma*sigma)) ))/(2* M_PI * pow(sigma,4));
}

Mat gaussianDerivateKernel(float sigma, bool direction = 0) {
	int size = ceil(sigma)*2+1;

	Mat kernel(size, size, CV_32F);


	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			kernel.at<float>(i,j) = (direction?gaussianDerivate(i-size/2,j-size/2,sigma):gaussianDerivate(j-size/2,i-size/2,sigma));
		}
	}

	return kernel;
}

void detectCorners(Mat &src, Mat &dst, float sigmaI, float sigmaD) {

    Mat Ix, Iy, Ixx;

    //Sobel(src, Ix, CV_32F, 1, 0, kernelSize);
    //Sobel(src, Iy, CV_32F, 0, 1, kernelSize);

    filter2D(src, Ix, CV_32F, gaussianDerivateKernel(sigmaD));
    filter2D(src, Iy, CV_32F, gaussianDerivateKernel(sigmaD,1));

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

    int kernelSize = ceil(sigmaI) * 2 + 1;
    Mat gaussKernel = getGaussianKernel(kernelSize, sigmaI);
    filter2D(covariance, covariance, covariance.depth(), gaussKernel);
    //boxFilter(covariance, covariance, covariance.depth(), Size(blockSize, blockSize));

    if(covariance.isContinuous() && covariance.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }

    for(int i=0 ; i < covariance.rows; i++ ) {
        for(int j = 0; j < covariance.cols; j++ ){

            Vec3f v = covariance.at<Vec3f>(i,j);
            v *= sigmaD * sigmaD;
            float a = v[0];
            float b = v[1];
            float c = v[2];
            dst.at<float>(i,j) = (float)(a*c - b*b - K*(a + c)*(a + c));
        }
    }


}

void harrisDetector(Mat &src, Mat &dst, float sigmaI) {

    Mat Ix, Iy, Ixx;

    //Sobel(src, Ix, CV_32F, 1, 0, kernelSize);
    //Sobel(src, Iy, CV_32F, 0, 1, kernelSize);

    filter2D(src, Ix, CV_32F, gaussianDerivateKernel(sigmaI*0.7));
    filter2D(src, Iy, CV_32F, gaussianDerivateKernel(sigmaI*0.7,1));
    

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
    
    imshow("Cov", covariance);
    
    GaussianBlur(covariance, covariance, Size(0,0), sigmaI);
    
    imshow("Cov 2", covariance);

	float min = FLT_MAX;
	float max = FLT_MIN;
    for(int i=0 ; i < covariance.rows; i++ ) {
        for(int j = 0; j < covariance.cols; j++ ){

            Vec3f v = covariance.at<Vec3f>(i,j);
            v *= sigmaI * sigmaI * 0.7 * 0.7;
            float a = v[0];
            float b = v[1];
            float c = v[2];
            
            dst.at<float>(i,j) = (float)(a*c - K*(a + c)*(a + c));
            if (max < dst.at<float>(i,j)) {
				cout << dst.at<float>(i,j) << ": " << a << "," << b << "," << c << endl;
			}
            
            min = (min < dst.at<float>(i,j)?min:dst.at<float>(i,j));
            max = (max < dst.at<float>(i,j)?dst.at<float>(i,j):max);
        }
    }
    cout << sigmaI << ": " << min << " " << max << endl;
    
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

    /*Mat g_k = gaussianDerivateKernel(1,1);

    cout << g_k << endl;*/

    Mat inputImage = cvLoadImageM(inputImageName.c_str());
    Mat grayImage(inputImage.rows, inputImage.cols, 0);

    cvtColor(inputImage, grayImage, CV_RGB2GRAY);
    imshow("Img", grayImage);
    waitKey();

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( grayImage.size(), CV_32FC1 );
    Mat corners = Mat::zeros( grayImage.size(), CV_32FC1 );
    /// Detector parameters
    
    for (float b=0; b<10; b++) {
		harrisDetector(grayImage, dst, pow(1.4,b));
		normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
		convertScaleAbs( dst_norm, dst_norm_scaled );
		imshow("Dst", dst_norm_scaled);
		
			int counter = 0, counter2 = 0;
			Mat outputImage = inputImage.clone();
			for( int j = 0; j < dst.rows ; j++ ) {
				for( int i = 0; i < dst.cols; i++ ) {
					float cur_point = (float) dst.at<float>(j,i);
					
					if (cur_point > THRESHOLD2) {
						circle(outputImage, Point(i,j), 5,  Scalar(255, 0, 0), 2, 8, 0 );
						counter++;
						
						bool condition = true;

						for (int m=-1; m<=1; m++) {
							for (int n=-1; n<=1; n++) {
								if (n == 0 && m == 0) continue;

								float neighbour = (float) dst.at<float>(j+m,i+n);

								if (neighbour >= cur_point) {
									condition = false;
									break;
								}
							}
							if (!condition) {
								break;
							}
						}
						
						if (condition) {
							counter2++;
							circle(outputImage, Point(i,j), 5,  Scalar(0, 255, 0), 2, 8, 0);
						}
					}
				}
			}
			cout << "Corners unfiltered: " << counter << endl;
			cout << "Corners filtered: " << counter2 << endl;
			
			imshow("Corners", outputImage );
			waitKey();
		}

    /*float apertureSize = 2.5;

    for (float b=3; b<10; b++) {


        //cout << "Scale: " << s << endl;
        detectCorners(grayImage, dst, b, apertureSize);
        //cornerHarris( grayImage, dst, b, apertureSize, k, BORDER_DEFAULT );

        /// Normalizing
        normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        convertScaleAbs( dst_norm, dst_norm_scaled );
        int counter = 0;
        Mat outputImage = inputImage.clone();
        for( int j = 0; j < dst_norm.rows ; j++ ) {
            for( int i = 0; i < dst_norm.cols; i++ ) {
                if( (int) dst_norm.at<float>(j,i) > THRESHOLD) {
                    circle(outputImage, Point(i, j), 5,  Scalar(255, 0, 0), 2, 8, 0 );
                    corners.at<float>(j,i) = dst_norm.at<float>(j,i);
                    counter++;
                }
            }
        }
        cout << "Corners unfiltered: " << counter << endl;

        counter = 0;
        for (int i=1; i<dst.rows-1; i++) {
            for (int j=1; j<dst.cols-1; j++) {
                float center = corners.at<float>(i,j);
                bool condition = true;

                for (int m=-1; m<=1; m++) {
                    for (int n=-1; n<=1; n++) {
                        if (n == 0 && m == 0) continue;

                        float neighbour = corners.at<float>(i+m,j+n);

                        if (neighbour >= center) {
                            condition = false;
                        }
                    }
                }

                if (condition) {
                    counter++;
                    circle(outputImage, Point(j, i), 5,  Scalar(0, 255, 0), 2, 8, 0);
                }
            }
        }
        /// Showing the result
        imshow("Corners", outputImage );
        cout << "Corners filtered: " << counter << endl;
        waitKey();
    }*/

    return EXIT_SUCCESS;
}
