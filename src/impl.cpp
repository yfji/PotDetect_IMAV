/*
 * impl.cpp
 *
 *      Author: JYF
 */

#include "PotDetect/impl.h"
#include <math.h>

#define PAD_LT	1
#define PAD_RB	10

#define NEAR_METRIC	100

void draw_bbox(Mat& frame, vector<Target>& targets, int width){
	if(targets.size()==0){
		//cout<<"no target detected"<<endl;
		return;
	}
	Scalar lineScalar=Scalar(0,255,255);
	Scalar fontScalar=Scalar(0,255,255);
	if(frame.channels()==1){	
		lineScalar=255;	
		fontScalar=255;
	}
	for(size_t i=0;i<targets.size();++i){
		Rect& loc=targets[i].location;
		if(loc.x<20 or loc.y<20)
			rectangle(frame, loc, lineScalar, width);
		else{
			int baseline;
			stringstream ss;
			ss<<targets[i].index<<":("<<loc.x<<","<<loc.y<<")";
			Size textSize=getTextSize(ss.str(), FONT_HERSHEY_PLAIN, 0.8, 1, &baseline);
			putText(frame, ss.str(), Point(loc.x, loc.y), FONT_HERSHEY_PLAIN, 0.8, fontScalar, 1, 8, false);
			
			line(frame, Point(loc.x,loc.y),Point(loc.x,loc.y+loc.height), lineScalar, width);
			if(textSize.width<loc.width)
				line(frame, Point(loc.x+textSize.width, loc.y), Point(loc.x+loc.width, loc.y), lineScalar, width);
			line(frame, Point(loc.x+loc.width, loc.y), Point(loc.x+loc.width, loc.y+loc.height), lineScalar, width);
			line(frame, Point(loc.x, loc.y+loc.height), Point(loc.x+loc.width, loc.y+loc.height), lineScalar, width);
		}
	}
}

void detect_bbox(Mat& src, vector<Target>& targets, bool bCurFrame){
	Mat scaled;
	int scale=4;
	
	resize(src, scaled, Size(src.cols/scale, src.rows/scale));
	
	feat_detect_saliency(scaled, SA_FT);
	//imshow("scaled", scaled);
	
	IplImage ipl=scaled;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq* pContour=NULL;
	extern double max_area;
	extern double min_area;
	extern vector<HogParam> params;
	extern MySVM hist_svm;
	extern MySVM hog_svm;

	max_area=(double)(src.cols*src.rows*0.5);
	min_area=81.0;
	if(targets.size()>0){
		Rect& max_loc=targets[0].location;
		Rect& min_loc=targets[targets.size()-1].location;
		max_area=(double)min(max_area, max_loc.height*max_loc.width*2.0);
		min_area=(double)max(min_area, min_loc.height*min_loc.width/2.0);
	}
	double ratio[2]={0.3,1.6};	//w/h
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<Target>::iterator iter=targets.begin();
	for(;iter!=targets.end();++iter){	(*iter).life-=1;	}

	for(;pContour;pContour=pContour->h_next){
		double area=fabs(cvContourArea(pContour))*scale*scale;
		if(area>max_area or area<min_area){
			cvSeqRemove(pContour,0);
			continue;
		}
		CvRect bbox=cvBoundingRect(pContour,0);
		Rect raw_bbox=Rect(bbox.x*scale, bbox.y*scale, bbox.width*scale, bbox.height*scale);
		double bw=1.0*raw_bbox.width;
		double bh=1.0*raw_bbox.height;
		if(bw/bh<ratio[0] or bw/bh>ratio[1]){
			cvSeqRemove(pContour,0);
			continue;
		}

		Mat roi=src(raw_bbox);
		int res_hog=feat_detect_svm(roi, hog_svm);
		int res_hist=1;//feat_detect_hist(roi, hist_svm);
		if(res_hog!=1 or res_hist!=1){
			cvSeqRemove(pContour,0);
			continue;
		}

		Target t;
		t.location=raw_bbox;
		int closest_index=0;
		bool is_new=is_new_target(t, targets, closest_index);

		if(is_new){
			if(not bCurFrame){
				if(target_detectable(t, src.size())){
					if(targets.size()<20){
						t.life=LIFE;
						t.index=targets.size();
						cout<<"target "<<t.index<<" added"<<endl;
						targets.push_back(t);
					}
					else{
						cout<<"too many pots"<<endl;
					}
				}
			}
		}
		else if(target_detectable(t, src.size())){	//update
			cout<<"target "<<targets[closest_index].index<<" updated"<<endl;
			targets[closest_index].location=raw_bbox;
			targets[closest_index].life=LIFE;
		}
		else{
			cout<<"target "<<targets[closest_index].index<<" disappeared"<<endl;
			vector<Target>::iterator iter_erase=targets.erase(targets.begin()+closest_index);
			for(;iter_erase!=targets.end();++iter_erase){
				(*iter_erase).index--;
			}
		}
		cvSeqRemove(pContour, 0);
	}
	if(targets.size()>0){
		vector<Target>::iterator iter=targets.begin();
		for(;iter!=targets.end();){
			if((*iter).life==0){
				iter=targets.erase(iter);
			}
			else{	++iter;	}
		}
	}
	vector<Target> temp=targets;
	vector<Target>().swap(targets);
	targets=temp;
	cvReleaseMemStorage(&pStorage);
}

void detect_bbox_kcf(Mat& src, vector<Target>& targets){
	//use kcf to track the target
}

bool is_new_target(Target& t, vector<Target>& targets, int& index){
	if(targets.size()==0)
		return true;
	bool is_new=true;
	int max_dist=10000;
	Rect& cur_loc=t.location;
	int metric=NEAR_METRIC;

	vector<Target>::iterator iter=targets.begin();
	for(;iter!=targets.end();++iter){
		Rect& loc=(*iter).location;
		if(abs(cur_loc.x-loc.x)>metric or abs(cur_loc.y-loc.y)>metric){
			continue;	//not near to this target
		}
		int dist=abs(cur_loc.x-loc.x)+abs(cur_loc.y-loc.y);
		if(dist<max_dist){
			max_dist=dist;
			index=iter-targets.begin();
		}
		is_new=false;
	}
	//find the nearest visible target
	return is_new;
}

bool target_detectable(Target& t, Size win){
	Rect& loc=t.location;
	
	if(loc.x>PAD_LT and loc.y>PAD_LT and loc.x+loc.width<win.width+PAD_RB and loc.y+loc.height<win.height+PAD_RB)
		return true;
	return false;
}

void track_by_detect(){

}

bool target_compare(Target& t1, Target& t2){
	int t1_area=t1.location.width*t1.location.height;
	int t2_area=t2.location.width*t2.location.height;
	return t1_area>t2_area;
}

