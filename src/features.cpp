/*
 * features.cpp
 *
 */

#include "PotDetect/features.h"
#include <math.h>
#include <fstream>
//#define min(x,y) x<y?x:y
using namespace cv;

void feat_max_blur(Mat& src, Size ksize){
	int h=src.rows;
	int w=src.cols;
	int pooled_x=w/ksize.width;
	int pooled_y=h/ksize.height;
	uchar* data=src.data;
	for(int y=0;y<pooled_y;++y){
		int ystart=y*ksize.height;
		for(int x=0;x<pooled_x;++x){
			int xstart=x*ksize.width;
			int max=-10000;
			for(int yy=ystart;yy<ystart+min(ksize.height,h-ystart);++yy){
				for(int xx=xstart;xx<xstart+min(ksize.width,w-xstart);++xx){
					int index=yy*w+xx;
					if(data[index]>max){
						max=data[index];
					}
				}
			}
			for(int yy=ystart;yy<ystart+min(ksize.height,h-ystart);++yy){
				for(int xx=xstart;xx<xstart+min(ksize.width,w-xstart);++xx){
					int index=yy*w+xx;
					data[index]=max;
				}
			}
		}
	}
}

void feat_detect_canny(Mat& src){
	src=feat_canny(src);
}

void feat_detect_houghCircle(Mat& src){
	src=feat_houghCircle(src);
}

Mat feat_detect_threshold(Mat& src){
	int ch=src.channels();
	Mat gray=src;
	if(ch==3)
		cvtColor(src,gray,CV_RGB2GRAY);
	int ksize=8;
	feat_max_blur(gray, Size(ksize,ksize));
	threshold(gray,gray,0,255,THRESH_OTSU);
	return gray;
}

void feat_detect_saliency(Mat& src, int mode){
	src=feat_saliency(src, mode);
}

MySVM get_svm(const char* feat){
	using namespace cv::ml;
	string svm_file_path="";
	if(strcmp(feat, "hog")==0){	
		char svm_file[30];
		sprintf(svm_file, "/home/yufeng/xmls/params/svm_hog_%d.xml", ID);
		svm_file_path=string(svm_file);	
	}
	else if(strcmp(feat, "hist")==0){	
		svm_file_path="/home/yufeng/xmls/svm_hist.xml";	
	}
	cout<<svm_file_path.c_str()<<endl;
	MySVM svm=ml::SVM::load(svm_file_path.c_str());
	return svm;
}

void get_logistic(Logistic& logi){
	string log_file_path="./log_pot.txt";
	logi.load(log_file_path.c_str());
}

int feat_detect_svm(Mat& src, MySVM& svm){
	Mat gray=src;
	if(src.channels()==3){
		cvtColor(src,gray,CV_BGR2GRAY);
	}
	extern vector<HogParam> params;
	
	int featLen=calcFeatLen(params);
	
	Mat sample(1,featLen,CV_32F);

	vector<float> hogFeat;

	getMSHogFeature(src, params, hogFeat);
	
	float* ptr=sample.ptr<float>(0);
	for(int k=0;k<featLen;++k)
		ptr[k]=hogFeat[k];
	float res=svm->predict(sample);
	return (int)res;
}

int feat_detect_hist(Mat& src, MySVM& svm){
	int featLen=(256/HIST_SEG);
	featLen=featLen*featLen*featLen;
	float* feat=new float[featLen];
	Mat sample(1,featLen,CV_32F);

	getHistogramFeature(src, feat);

	float* ptr=sample.ptr<float>(0);
	for(int k=0;k<featLen;++k)
		ptr[k]=feat[k];
	float res=svm->predict(sample);
	delete feat;
	return (int)res;
}

float feat_detect_logistic(Mat& src, Logistic& logi){
	Mat gray=src;
		if(src.channels()==3){
			cvtColor(src,gray,CV_BGR2GRAY);
		}
		int fixed_size=SCALED_SIZE;
		int stride=STRIDE;
		int cellSize=CELL_SIZE;
		int blockSize=BLOCK_SIZE;
		int featLen=((fixed_size-blockSize)/stride+1)\
				*((fixed_size-blockSize)/stride+1)*36;

		Mat sample(1,featLen,CV_32F);
		vector<float> hogFeat;
		resize(gray, gray, Size(fixed_size, fixed_size));
		normalize(gray, gray, 0,255,NORM_MINMAX);
		feat_hog_cv2(gray, Size(blockSize, blockSize), Size(cellSize, cellSize),\
											Size(stride,stride), hogFeat);
		float* ptr=sample.ptr<float>(0);
		for(int k=0;k<featLen;++k)
			ptr[k]=hogFeat[k];
		normalize(sample,sample,0,1,NORM_MINMAX);
		int res=logi.classify(sample);
		return res;
}

void feat_train_svm(int iter){
	int fixed_size=SCALED_SIZE;
	int stride=STRIDE;
	int cellSize=CELL_SIZE;
	int blockSize=BLOCK_SIZE;
	int featLen=((fixed_size-blockSize)/stride+1)\
			*((fixed_size-blockSize)/stride+1)*36;
	string label_file_path="/mnt/I/TestOpenCV/Videos/pot_train/label_ubuntu.txt";
	char svm_file[30];
	sprintf(svm_file, "./params/svm_hog_%d", ID);
	string svm_file_path=string(svm_file);
	ifstream in;
	in.open(label_file_path.c_str(), ios::in);
	vector<float> hogFeat;
	int pos_cnt=0, neg_cnt=0;
	cout<<"preparing data..."<<endl;
	while(not in.eof()){
		string line;
		int label;
		in>>line;
		in>>label;
		if(line.length()<=1)	continue;
		if(label==1)	++pos_cnt;
		else if(label==-1)	++neg_cnt;
		else	assert(label==1 or label==-1);
	}
	cout<<pos_cnt<<", "<<neg_cnt<<endl;
	Mat sampleMat(pos_cnt+neg_cnt, featLen, CV_32F);
	Mat labelMat(pos_cnt+neg_cnt, 1, CV_32S);
	int r=0;
	in.close();
	in.open(label_file_path.c_str(), ios::in);
	Mat gray;
	while(not in.eof()){
	//for(int i=0;i<10;++i){
		string line;
		int label;
		in>>line;
		in>>label;
		if(line.length()<=1)
			continue;
		cout<<line<<", "<<label<<endl;
		Mat image=imread(line);
		resize(image, image, Size(fixed_size, fixed_size));
		if(image.channels()==3)
			cvtColor(image,gray,CV_BGR2GRAY);
		else
			gray=image;
		normalize(gray,gray,0,255,NORM_MINMAX);
		feat_hog_cv2(image, Size(blockSize, blockSize), Size(cellSize, cellSize),\
									Size(stride,stride), hogFeat);
		assert(hogFeat.size()==featLen);
		float* featPtr=sampleMat.ptr<float>(r);
		float* labelPtr=labelMat.ptr<float>(r);
		for(int k=0;k<featLen;++k)
			featPtr[k]=hogFeat[k];
		//Mat row=sampleMat.row(r);
		//normalize(row, row,0,1,NORM_MINMAX);
		labelPtr[0]=label;
		++r;
	}
	in.close();
	cout<<"data prepared, start training..."<<endl;
	using namespace ml;
	MySVM svm=SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	//TermCriteria tc(CV_TERMCRIT_ITER, iter, FLT_EPSILON);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, iter, FLT_EPSILON));
	svm->train(sampleMat, ROW_SAMPLE, labelMat);
	svm->save(svm_file_path.c_str());
	cout<<"finished"<<endl;
}

void train_svm_hist(int iter){
	int featLen=256/HIST_SEG;
	featLen=featLen*featLen*featLen;
	string label_file_path="I:/TestOpenCV/Videos/pot_train/label_lab.txt";
	string svm_file_path="./svm_hist.xml";
	ifstream in;
	in.open(label_file_path.c_str(), ios::in);

	float* feat=new float[featLen];
	int pos_cnt=0, neg_cnt=0;
	cout<<"preparing data..."<<endl;
	while(not in.eof()){
		string line;
		int label;
		in>>line;
		in>>label;
		if(line.length()<=1)	continue;
		if(label==1)	++pos_cnt;
		else if(label==-1)	++neg_cnt;
		else	assert(label==1 or label==-1);
	}
	cout<<pos_cnt<<", "<<neg_cnt<<endl;
	Mat sampleMat(pos_cnt+neg_cnt, featLen, CV_32F);
	Mat labelMat(pos_cnt+neg_cnt, 1, CV_32S);
	int r=0;
	in.close();
	in.open(label_file_path.c_str(), ios::in);

	while(not in.eof()){
		string line;
		int label;
		in>>line;
		in>>label;
		if(line.length()<=1)
			continue;
		Mat image=imread(line);

		getHistogramFeature(image, feat);

		float* featPtr=sampleMat.ptr<float>(r);
		float* labelPtr=labelMat.ptr<float>(r);
		for(int k=0;k<featLen;++k)
			featPtr[k]=feat[k];
		//Mat row=sampleMat.row(r);
		//normalize(row, row,0,255,NORM_MINMAX);
		labelPtr[0]=label;
		++r;
	}
	in.close();
	cout<<"data prepared, start training..."<<endl;
	using namespace ml;
	MySVM svm=SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, iter, FLT_EPSILON));
	svm->train(sampleMat, ROW_SAMPLE, labelMat);
	svm->save(svm_file_path.c_str());
	cout<<"finished"<<endl;
	delete feat;
}

void feat_train_logistic(int iter){
	int fixed_size=SCALED_SIZE;
	int stride=STRIDE;
	int cellSize=CELL_SIZE;
	int blockSize=BLOCK_SIZE;
	int featLen=((fixed_size-blockSize)/stride+1)\
			*((fixed_size-blockSize)/stride+1)*36;
	string label_file_path="I:/TestOpenCV/Videos/pot_train/label.txt";
	string log_file_path="./log_pot.txt";
	ifstream in;
	in.open(label_file_path.c_str(), ios::in);
	vector<float> hogFeat;
	int pos_cnt=0, neg_cnt=0;
	cout<<"preparing data..."<<endl;
	while(not in.eof()){
		string line;
		int label;
		in>>line;
		in>>label;
		if(line.length()<=1)	continue;
		if(label==1)	++pos_cnt;
		else if(label==-1)	++neg_cnt;
		else	assert(label==1 or label==-1);
	}
	cout<<pos_cnt<<", "<<neg_cnt<<endl;
	Mat sampleMat(pos_cnt+neg_cnt, featLen, CV_32F);
	Mat labelMat(1, pos_cnt+neg_cnt, CV_32F);
	int r=0;
	in.close();
	in.open(label_file_path.c_str(), ios::in);
	Mat gray;
	float* labelPtr=labelMat.ptr<float>(0);
	while(not in.eof()){
	//for(int i=0;i<10;++i){
		string line;
		int label;
		in>>line;
		in>>label;
		Mat image=imread(line);
		if(line.length()<=1)
			continue;
		resize(image, image, Size(fixed_size, fixed_size));
		if(image.channels()==3)
			cvtColor(image,gray,CV_BGR2GRAY);
		else
			gray=image;
		normalize(gray,gray,0,255,NORM_MINMAX);
		feat_hog_cv2(image, Size(blockSize, blockSize), Size(cellSize, cellSize),\
									Size(stride,stride), hogFeat);
		assert(hogFeat.size()==featLen);
		float* featPtr=sampleMat.ptr<float>(r);

		for(int k=0;k<featLen;++k)
			featPtr[k]=hogFeat[k];
		Mat row=sampleMat.row(r);
		normalize(row, row,0,1,NORM_MINMAX);
		if(label==-1){	label=0;	}
		labelPtr[r]=(float)label;
		++r;
	}
	in.close();
	cout<<"data prepared, start training..."<<endl;
	Logistic logi(featLen);
	logi.lr=0.001;
	logi.minibatch=10;
	logi.step=10;
	logi.gamma=0.1;
	logi.train(sampleMat, labelMat, iter);
	logi.save(log_file_path.c_str());
}