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
#define K 0.06
#define THRESHOLD_LAPLACE 10.0
#define THRESHOLD_HARRIS 1500.0
#define S 0.7
#define epsilon 1.4

#define S0 1.5
#define COUNTDOWN 3

#define LAPLACE_S_STEP 0.1
#define LAPLACE_S_START 0.7
#define LAPLACE_S_COUNT 8

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

void harrisDetector(Mat &src, Mat &dst, float sigmaI, float *maxR) {

    Mat Ix, Iy;

    //Sobel(src, Ix, CV_32F, 1, 0, kernelSize);
    //Sobel(src, Iy, CV_32F, 0, 1, kernelSize);

    filter2D(src, Ix, CV_32F, gaussianDerivateKernel(sigmaI * S));
    filter2D(src, Iy, CV_32F, gaussianDerivateKernel(sigmaI * S, 1));


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

    GaussianBlur(covariance, covariance, Size(0,0), sigmaI);

	float min = FLT_MAX;
	float max = FLT_MIN;
    for(int i=0 ; i < covariance.rows; i++ ) {
        for(int j = 0; j < covariance.cols; j++ ){

            Vec3f v = covariance.at<Vec3f>(i,j);
            v *= sigmaI * sigmaI * S * S;
            float a = v[0];
            float b = v[1];
            float c = v[2];

            dst.at<float>(i,j) = (float)(a*c -b*b- K*(a + c)*(a + c));

            min = (min < dst.at<float>(i,j)?min:dst.at<float>(i,j));
            max = (max < dst.at<float>(i,j)?dst.at<float>(i,j):max);
        }
    }
    cout << sigmaI << ": " << min << " " << max << endl;

    *maxR = max;
}

float getLaplacian(Mat &src, int x, int y, float sigma0, float *maxR) {
	Mat Ixx, Iyy;
    
    float max = FLT_MIN;
    
    float curLoG;
    float curSigma;
    
    for (int i = 0; i < LAPLACE_S_COUNT; i++) {
		curSigma = sigma0 * (LAPLACE_S_START + LAPLACE_S_STEP * i);
		
		filter2D(src, Ixx, CV_32F, gaussianDerivateKernel(curSigma));
		filter2D(Ixx, Ixx, CV_32F, gaussianDerivateKernel(curSigma));
		filter2D(src, Iyy, CV_32F, gaussianDerivateKernel(curSigma, 1));
		filter2D(Iyy, Iyy, CV_32F, gaussianDerivateKernel(curSigma, 1));
		
		curLoG = curSigma*curSigma*abs(Ixx.at<float>(x,y)+Iyy.at<float>(x,y));
		if (max < curLoG) {
			max = curLoG;
			*maxR = curSigma;
		}
	}
    
    return max;
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
    Mat cornerScales = Mat::zeros( grayImage.size(), CV_32FC1 );
    Mat corners = Mat::zeros( grayImage.size(), CV_32FC1 );
    /// Detector parameters

	Mat outputImage = inputImage.clone();
	float max;
	float s = 1.5;
	int countdown = COUNTDOWN;
    do {
		harrisDetector(grayImage, dst, s, &max);
		//normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
		//convertScaleAbs( dst_norm, dst_norm_scaled );
		//imshow("Dst", dst_norm_scaled);

			int counter = 0, counter2 = 0;
			for( int j = 0; j < dst.rows ; j++ ) {
				for( int i = 0; i < dst.cols; i++ ) {
					float cur_point = (float) dst.at<float>(j,i);

					if (cur_point > THRESHOLD_HARRIS) {
						//circle(outputImage, Point(i,j), 5,  Scalar(255, 0, 0), 1, 8, 0 );
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
							cornerScales.at<float>(j,i) = s;

							for (int m=-1; m<=1; m++) {
								for (int n=-1; n<=1; n++) {
									if (n == 0 && m == 0) continue;

									//cornerScales.at<float>(j+m,i+n) = 0;
								}
							}
						}
					}
				}
			}
			cout << "Corners unfiltered: " << counter << endl;
			cout << "Corners filtered: " << counter2 << endl;
		s *= epsilon;
		if (max > THRESHOLD_HARRIS) {
			countdown = COUNTDOWN;
		} else {
			countdown -= 1;
		}
		cout << max << endl;
	} while (countdown > 0);
		
		float Laplaci;
		float sigmaX;
	for( int j = 0; j < cornerScales.rows ; j++ ) {
		for( int i = 0; i < cornerScales.cols; i++ ) {
			if (cornerScales.at<float>(j,i) > 0) {
				Laplaci = getLaplacian(grayImage, j, i, cornerScales.at<float>(j,i), &sigmaX);
				if (Laplaci > THRESHOLD_LAPLACE) {
					cout << i << "," << j << ": " << cornerScales.at<float>(j,i) << " vs " << Laplaci << endl;
					
					circle(outputImage, Point(i,j), 3*sigmaX,  Scalar(255-(int)3*3*sigmaX, 0, (int)3*3*sigmaX), 2, 8, 0 );
				}
			}
		}
	}

	namedWindow("Corners", CV_WINDOW_NORMAL);
	imshow("Corners", outputImage );
	waitKey();

    return EXIT_SUCCESS;
}
