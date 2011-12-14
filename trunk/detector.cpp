#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
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

using namespace std;
using namespace cv;

struct ScaledPoint {
	float scale;
	Point point;
};

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

void getHarrisLaplace(Mat &src, float sigma0, map<int, Point> harrisPoints, map<int, ScaledPoint> &harrisLaplacePoints) {
	Mat DefIxx[LAPLACE_S_COUNT], DefIyy[LAPLACE_S_COUNT];
	
    float sigma;
	for (int i = 0; i < LAPLACE_S_COUNT; i++) {
		sigma = sigma0 * (LAPLACE_S_START + LAPLACE_S_STEP * i);
		
		filter2D(src, DefIxx[i], CV_32F, gaussianDerivateKernel(sigma));
		filter2D(DefIxx[i], DefIxx[i], CV_32F, gaussianDerivateKernel(sigma));
		filter2D(src, DefIyy[i], CV_32F, gaussianDerivateKernel(sigma, 1));
		filter2D(DefIyy[i], DefIyy[i], CV_32F, gaussianDerivateKernel(sigma, 1));
	}
	
    map<int,Point>::iterator it;

    for (it = harrisPoints.begin(); it != harrisPoints.end(); ++it) {
		int xK0 = (it->second).x;
		int yK0 = (it->second).y;
		bool condition = true;
		int xK1 = xK0, yK1 = yK0, iK;
		float sigmaK0 = sigma0, sigmaK1;
		float curLoG, curSigma;
		float LoG = FLT_MIN;
		Mat Ixx, Iyy;
		Mat curIxx(DefIxx[0].size(), DefIxx[0].type());
		Mat curIyy(DefIyy[0].size(), DefIyy[0].type());
		
		for (int i = 0; i < LAPLACE_S_COUNT; i++) {
			curSigma = sigma0 * (LAPLACE_S_START + LAPLACE_S_STEP * i);
			
			curLoG = curSigma * curSigma * abs(DefIxx[i].at<float>(yK0,xK0) + DefIyy[i].at<float>(yK0,xK0));
			if (LoG < curLoG) {
				LoG = curLoG;
				sigmaK1 = curSigma;
				iK = i;
			}
		}
		
		if (LoG < THRESHOLD_LAPLACE) {
			continue;
		}
		
		
		for (int m=-1; m<=1; m++) {
			for (int n=-1; n<=1; n++) {
				if (n == 0 && m == 0) continue;

				curLoG = sigmaK1 * sigmaK1 * abs(DefIxx[iK].at<float>(yK0+m,xK0+n) + DefIyy[iK].at<float>(yK0+m,xK0+n));

				if (LoG < curLoG) {
					LoG = curLoG;
					xK1 = xK0+n;
					yK1 = yK0+m;
				}
			}
		}
		
		while (xK1 != xK0 || yK1 != yK0 || sigmaK0 != sigmaK1) {
			cout << "X[" << xK1 << "," << yK1 << "]: LoG " << LoG << ", Sigma " << sigmaK1 << endl;
			xK0 = xK1;
			yK0 = yK1;
			sigmaK0 = sigmaK1;
		
			for (int i = 0; i < LAPLACE_S_COUNT; i++) {
				curSigma = sigmaK0 * (LAPLACE_S_START + LAPLACE_S_STEP * i);
			
				filter2D(src, Ixx, CV_32F, gaussianDerivateKernel(curSigma));
				filter2D(Ixx, Ixx, CV_32F, gaussianDerivateKernel(curSigma));
				filter2D(src, Iyy, CV_32F, gaussianDerivateKernel(curSigma, 1));
				filter2D(Iyy, Iyy, CV_32F, gaussianDerivateKernel(curSigma, 1));
				
				curLoG = curSigma * curSigma * abs(Ixx.at<float>(yK0,xK0) + Iyy.at<float>(yK0,xK0));
				if (LoG < curLoG) {
					LoG = curLoG;
					sigmaK1 = curSigma;
					curIxx = Ixx.clone();
					curIyy = Iyy.clone();
				}
			}
			
			for (int m=-1; m<=1; m++) {
				for (int n=-1; n<=1; n++) {
					if (n == 0 && m == 0) continue;

					curLoG = sigmaK1 * sigmaK1 * abs(curIxx.at<float>(yK0+m,xK0+n) + curIyy.at<float>(yK0+m,xK0+n));

					if (LoG < curLoG) {
						LoG = curLoG;
						xK1 = xK0+n;
						yK1 = yK0+m;
					}
				}
			}
		}
		
		harrisLaplacePoints[xK1 + curIxx.rows * yK1].scale = sigmaK1;
		harrisLaplacePoints[xK1 + curIxx.rows * yK1].point = Point(xK1, yK1);
		cout << "O[" << xK1 << "," << yK1 << "]: LoG " << LoG << ", Sigma " << sigmaK1 << endl;
    }
}

int main(int argc, char* argv[]) {

    /*map<float, map<int, Point> > points;

    float a = 0.0f;
    int x = 1, y = 0;
    int b = 0;
    points[a] = map<int, Point>();

    //b = x + width * y;

    points[a][b] = Point(x,y);

    map<int,Point>::iterator it;

    for(it = points[a].begin(); it != points[a].end(); ++it) {
        cout << (it->second).x << endl;
    }
    */

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


	map<float, map<int, Point> > harrisPoints;
	map<int, ScaledPoint> harrisLaplacePoints;
	Mat outputImage = inputImage.clone();
	float max;
	float s = S0;
	int countdown = COUNTDOWN;
    do {
		harrisPoints[s] = map<int, Point>();
		
		harrisDetector(grayImage, dst, s, &max);
		//normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
		//convertScaleAbs( dst_norm, dst_norm_scaled );
		//imshow("Dst", dst_norm_scaled);

			int counter = 0, counter2 = 0;
			for( int j = 0; j < dst.rows ; j++ ) {
				for( int i = 0; i < dst.cols; i++ ) {
					float cur_point = (float) dst.at<float>(j,i);

					if (cur_point > THRESHOLD_HARRIS) {
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
							harrisPoints[s][i + dst.rows * j] = Point(i, j);
							

							//for (int m=-1; m<=1; m++) {
							//	for (int n=-1; n<=1; n++) {
							//		if (n == 0 && m == 0) continue;

									//cornerScales.at<float>(j+m,i+n) = 0;
							//	}
							//}
						}
					}
				}
			}
			cout << "Corners unfiltered: " << counter << endl;
			cout << "Corners filtered: " << counter2 << endl;

		getHarrisLaplace(grayImage, s, harrisPoints[s], harrisLaplacePoints);
		
		s *= epsilon;
		if (max > THRESHOLD_HARRIS) {
			countdown = COUNTDOWN;
		} else {
			countdown -= 1;
		}
		cout << max << endl;
	} while (countdown > 0);
	
	
    map<int,ScaledPoint>::iterator it;

    for(it = harrisLaplacePoints.begin(); it != harrisLaplacePoints.end(); ++it) {
		circle(outputImage, (it->second).point, 3*(it->second).scale,  Scalar(255-(int)3*3*(it->second).scale, 0, (int)3*3*(it->second).scale), 2, 8, 0 );
    }

	/*	float Laplaci;
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
	}*/

	namedWindow("Corners", CV_WINDOW_NORMAL);
	imshow("Corners", outputImage );
	waitKey();

    return EXIT_SUCCESS;
}
