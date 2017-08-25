/*
 * extract_feat.cpp
 */

#include "PotDetect/extract_feat.h"
#include "PotDetect/saliency.h"

Mat feat_canny(Mat& src){
	int ch=src.channels();
	Mat gray=src;
	if(ch==3){
		cvtColor(src, gray, CV_RGB2GRAY);
	}
	Mat canny;
	Canny(src, canny, 100,300);
	return canny;
}

Mat feat_houghCircle(Mat& src){
	int ch=src.channels();
	Mat gray=src;
	if(ch==3){
		cvtColor(src, gray, CV_RGB2GRAY);
	}
	GaussianBlur(gray, gray, Size(3,3),0.1);
	int radiusThresh=0;
	Mat canvas=Mat::zeros(src.size(), CV_8UC1);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1.5 ,10, 80, 110, 35 ,65);
    vector<Vec3f>::iterator iter=circles.begin();
    for(;iter!=circles.end();++iter){
    	Vec3f point=*iter;
    	int centerX=point[0];
    	int centerY=point[1];
    	int radius=point[2];
    	if(radius>radiusThresh){
    		circle(canvas, Point(centerX,centerY),radius,255);
    	}
    }
    return canvas;
}


void feat_hog_cv2(Mat& src, Size blockSize, Size cellSize, Size stride, vector<float>& hogFeat){
	int ch=src.channels();
	Mat gray=src;
	if(ch==3){
		cvtColor(src, gray, CV_RGB2GRAY);
	}
	HOGDescriptor hog;
	hog.blockSize=blockSize;
	hog.cellSize=cellSize;
	hog.blockStride=stride;
	hog.winSize=Size(src.cols,src.rows);
	hog.compute(src, hogFeat);
}

void getHogFeature(Mat& image, HogParam& param, vector<float>& hogFeat, bool cvt){
	Mat gray=image;
	if(cvt){
		int ch=image.channels();
		if(ch==3){
			cvtColor(image, gray, CV_BGR2GRAY);
		}
		resize(gray, gray, param.winSize);
		normalize(gray, gray, 0, 255, NORM_MINMAX);
	}
	HOGDescriptor hog;
	hog.blockSize=Size(param.blockSize, param.blockSize);
	hog.cellSize=Size(param.cellSize, param.cellSize);
	hog.blockStride=Size(param.stride, param.stride);
	hog.winSize=param.winSize;
	hog.compute(image, hogFeat);
}

void getMSHogFeature(Mat& image, vector<HogParam>& params, vector<float>& msHogFeat){
	Mat gray=image;
	int ch=image.channels();
	if(ch==3){
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	resize(gray, gray, params[0].winSize);
	normalize(gray, gray, 0, 255, NORM_MINMAX);
	for(unsigned int i=0;i<params.size();++i){
		HogParam& param=params[i];
		vector<float> ssHogFeat;
		getHogFeature(gray, param, ssHogFeat, false);
		for(unsigned int j=0;j<ssHogFeat.size();++j)
			msHogFeat.push_back(ssHogFeat[j]);
	}
}

void getHistogramFeature(Mat& image, float*& feat){
	resize(image, image, Size(HIST_SIZE, HIST_SIZE));
	int h=image.rows;
	int w=image.cols;
	int stride=w*3;

	int total=256/HIST_SEG;
	total=total*total*total;
	uchar* data=image.data;
	if(feat==NULL){	feat=new float[total];	}
	for(int i=0;i<total;++i){	feat[i]=0.0;	}
	for(int i=0;i<h*stride;i+=3){
		int b=(int)data[i];
		int g=(int)data[i+1];
		int r=(int)data[i+2];

		b/=HIST_SEG;
		g/=HIST_SEG;
		r/=HIST_SEG;

		feat[b*256+g*16+r]+=1.0;
	}
	Normalize<float>(feat, total, 0.0f, 1.0f);
}

Mat feat_hog(Mat& src){
	IplImage _ipl=src;
	CvLSVMFeatureMapCaskade * map;
	int cellSize=8;
	getFeatureMaps(&_ipl,cellSize,&map);
	normalizeAndTruncate(map,0.2f);
	PCAFeatureMaps(map);
	Mat FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
	FeaturesMap = FeaturesMap.t();
	freeFeatureMapObject(&map);
	return FeaturesMap;
}

Mat feat_saliency(Mat& src, int mode){
	Mat sa;
	if(mode==SA_FREQ){
		int ch=src.channels();
		Mat gray=src;
		if(ch==3){
			cvtColor(src, gray, CV_RGB2GRAY);
		}
		detectSaliency(gray, sa);
	}

	else if(mode==SA_FT){
		saliencyDetectFT(src, sa);
	}
	else
		assert(0);
	threshold(sa, sa, 0, 255, THRESH_OTSU);
	return sa;
}
