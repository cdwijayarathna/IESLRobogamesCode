#include "StdAfx.h"
#include "cv.h"
#include "highgui.h"
#include <math.h>
#include <queue>
#include <deque>

using namespace cv;

IplImage* g_image = NULL;
IplImage* g_gray = NULL;
int g_thresh = 100;
CvMemStorage* g_storage = NULL;
vector<CvPoint> OSurround;
queue<CvPoint> roadPoints;
int robot=0;

void getSampleAdjPoints(int r){
	IplImage* image = cvCreateImage(cvSize(100,100),IPL_DEPTH_8U,1);
	cvCircle(image,cvPoint(50,50),r,cvScalar(0,0,0,0),1,8,0);	
	cvFloodFill(image,cvPoint(50,50),CV_RGB(0,0,0));
	OSurround = vector<CvPoint>();
	for( int y=0; y<image->height; y++ ) 
	{
		uchar* ptr = (uchar*) (image->imageData + y * image->widthStep);
		for( int x=0; x<image->width; x++ ) 
		{
			if(	ptr[x] <100 )
			{
				OSurround.push_back(cvPoint(x-50,y-50));				
			}			
		}
	}	
}

struct Triangle{
	CvPoint center;
	CvPoint head;
	CvPoint tail1;
	CvPoint tail2;
	CvPoint midPoint;
	
	double area;
};

CvPoint centroid(CvSeq* contour) {
CvPoint pt, prev_pt, centroid = {0,0};
CvSeqReader reader;

cvStartReadSeq( contour, &reader, 0 );

int count = contour->total;
count -= !CV_IS_SEQ_CLOSED(contour);

CV_READ_SEQ_ELEM( prev_pt, reader );
int area=0;
for( int i = 0; i < count; ++i ) {
CV_READ_SEQ_ELEM( pt, reader );

int moment = prev_pt.x * pt.y - pt.x * prev_pt.y;

area += moment;
centroid.x += (pt.x + prev_pt.x) * moment;
centroid.y += (pt.y + prev_pt.y) * moment;
prev_pt = pt;
}

area *= 0.5;
centroid.x /= 6.0 * area;
centroid.y /= 6.0 * area;

return centroid;
}

double getLength(CvPoint p1,CvPoint p2){
	//if(p1==NULL||p2==NUll)
		//return 0;
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+0.0);
}

CvSeq* getAllContours(){
	if( g_storage==NULL ) {
	g_gray = cvCreateImage( cvGetSize(g_image), 8, 1 );
	g_storage = cvCreateMemStorage(0);
	} else {
		cvClearMemStorage( g_storage );
	}
	CvSeq* contours = 0;
	cvCvtColor( g_image, g_gray, CV_BGR2GRAY );
	cvThreshold( g_gray, g_gray, 105, 255, CV_THRESH_BINARY );
	cvFindContours( g_gray, g_storage, &contours,sizeof(CvContour),CV_RETR_LIST);
	
	cvZero( g_gray );
	//if( contours )
	cvDrawContours(g_gray,contours,cvScalarAll(255),cvScalarAll(255),100);	
	return contours;
}

vector<Triangle> getTriangles(CvSeq* contours){
	vector<Triangle> tringles = vector<Triangle>();
	
	while(contours=contours->h_next){
		double area=cvContourArea(contours);		
		if(area<600.0&&area>400){
			
			Triangle tri=Triangle();
			tri.area=area;
			CvPoint2D32f point;
			point=cvPoint2D32f(0.0,0.0);
			float radious=0;
			cvMinEnclosingCircle(contours,&point,&radious);
			//tri.center=cvPoint(point.x,point.y);
			tri.center = centroid(contours);
			vector<CvPoint> points;
			vector<CvPoint> temp_points;
			points = vector<CvPoint>();
			temp_points = vector<CvPoint>();
			CvPoint* t_point;
			int x=0;
			int y=0;
			for(int i=0;i<contours->total;i++){
				t_point=(CvPoint*)cvGetSeqElem(contours,i);
				x=t_point->x;
				y=t_point->y;
				
				if(fabs(sqrt((x-point.x)*(x-point.x)+(y-point.y)*(y-point.y))-radious)<0.8){
					points.push_back(*t_point);
				}
			}
			for(int i=0;i<points.size();i++){

				if(points.at(i).x==-1){
					continue;	
				}
				for(int j=i+1;j<points.size();j++){
					if(fabs(points.at(i).x-points.at(j).x+0.0)<5&&fabs(points.at(i).y-points.at(j).y+0.0)<5){
						points.at(i).x=(points.at(j).x+points.at(i).x)/2;
						points.at(i).y=(points.at(j).y+points.at(i).y)/2;
						points.at(j).x=-1;
					}
				}
				temp_points.push_back(points.at(i));				
			}			
			if(temp_points.size()==3){
			double l[3];
			
			l[0]=getLength(temp_points.at(1),temp_points.at(2));
			l[1]= getLength(temp_points.at(0),temp_points.at(2));
			l[2]= getLength(temp_points.at(0),temp_points.at(1));
			
			double minLength=l[0];
			int edge=0;
				for(int i=1;i<3;i++){
					if(l[i]<minLength){
						minLength=l[i];
						edge=i;
					}
				}
				CvPoint midPoint;
				tri.head=temp_points.at(edge);
				if(edge==0){					
					tri.tail1=temp_points.at(1);
					tri.tail2=temp_points.at(2);
					tri.midPoint = cvPoint((temp_points.at(1).x+temp_points.at(2).x)/2,(temp_points.at(1).y+temp_points.at(2).y)/2);

				}else if(edge==1){
					tri.tail1=temp_points.at(0);
					tri.tail2=temp_points.at(2);		
					tri.midPoint = cvPoint((temp_points.at(0).x+temp_points.at(2).x)/2,(temp_points.at(0).y+temp_points.at(2).y)/2);
				}else{
					tri.tail1=temp_points.at(0);
					tri.tail2=temp_points.at(1);		
					tri.midPoint = cvPoint((temp_points.at(1).x+temp_points.at(0).x)/2,(temp_points.at(1).y+temp_points.at(0).y)/2);
				}
				tringles.push_back(tri);
			}
		}
	}
	return tringles;
}

CvPoint getBall(CvSeq* contours){
	CvPoint point=cvPoint(-1,-1);
	while(contours=contours->h_next){
		double area=cvContourArea(contours);			
		if(area<200.0&&area>25){
			point = centroid(contours);
			CvPoint2D32f tempPoint;
			int x=0,y=0,count=0;
			for(int i=0;i<OSurround.size();i++){
				tempPoint = cvPoint2D32f(point.x+OSurround.at(i).x,point.y+OSurround.at(i).y);
				
				double val=cvPointPolygonTest(contours,tempPoint,1);
				if(val>=0){
					x+=tempPoint.x;
					y+=tempPoint.y;
					count++;
				cvCircle(g_gray,cvPoint(tempPoint.x,tempPoint.y),1,cvScalarAll(255));
				}
			}
			point.x=x/count;
			point.y=y/count;
		}
	}
	if(point.x>0)
	roadPoints.push(point);
	return point;
}

void cvt() {
	CvSeq* contours = getAllContours();
	vector<Triangle> triangles = getTriangles(contours);
	for(int i=0;i<triangles.size();i++){
		cvCircle(g_gray,triangles.at(i).head,5,cvScalarAll(255));
		cvCircle(g_gray,triangles.at(i).center,10,cvScalarAll(255));
		cvLine(g_gray,triangles.at(i).head,triangles.at(i).midPoint,cvScalarAll(255));
	}
	CvPoint ball = getBall(contours);
	if(ball.x!=-1)
	cvCircle(g_gray,ball,10,cvScalarAll(255));
	cvShowImage( "Contours", g_gray );	
	
}

vector<Vec2f> getBox(vector<Vec2f> lines)
{
	for(int i=0;i<lines.size();i++){
		float theta = (float)(lines[i][1]*180/CV_PI);
		float p =(lines[i][0]);
		if(p<0){
			theta=theta-180;
			
			p=p*(-1);
		}
		lines[i][0]=p;
		lines[i][1]=theta*CV_PI/180;
		if(p<8&&theta<2){
			lines.erase(lines.begin()+i);
		}
	}
	vector<Vec2f> temp;
	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	for(int i=0;i<lines.size();i++)
	{
		if(lines[i][0]!=0){
		float test_p=lines[i][0];
		float test_t=lines[i][1]*180/CV_PI;
		printf("aceesing line %d    %f   %f\n",i,lines[i][0],lines[i][1]);
		//temp.push_back(lines[i]);
		float add_p=lines[i][0];
		float add_t=lines[i][1];
		int count=1;
		for(int j=0;j<lines.size();j++)
		{
			if(i==j)continue;
			float deg=lines[j][1]*180/CV_PI;
			if( ((lines[j][0]>=lines[i][0]-20) && (lines[j][0]<=lines[i][0]+20)) && (deg>=test_t-5) && (deg<=test_t+5) && lines[j][0]!=0)
			{
				//temp.push_back(lines[j]);
				add_p+=lines[j][0];
				add_p/=2;
				add_t+=lines[j][1];add_t/=2;
				count++;printf("		count %d    %f   %f\n",j,lines[j][0],deg);
				lines[j][0]=0;
				lines[j][1]=0;
				
				//lines.erase(lines.begin()+i+j);
				//j--;
			}

		}
		//add_p=add_p/count;
		//add_t=add_t/count;
		//=add_p;
		//lines[i][1]=add_t/;
		//add_t=add_t*CV_PI/180;
		Vec2f tem=Vec2f(add_p,add_t);
		temp.push_back(tem);printf("     temp    %f   %f\n",add_p,add_t);
		}}
	return temp;
}

vector<Vec2f> processGrid(IplImage* image){
	Mat mat = Mat(image,1);
	Mat sudoku;
	cvtColor(mat,sudoku,CV_RGB2GRAY);
	Mat outerBox = Mat(sudoku.size(), CV_8UC1);
	GaussianBlur(sudoku, sudoku, Size(11,11), 0);
   
	adaptiveThreshold(sudoku, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	bitwise_not(outerBox, outerBox);
	Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
   
	dilate(outerBox, outerBox, kernel);
	int count=0;
    int max=-1;
    Point maxPt;
 
    for(int y=0;y<outerBox.size().height;y++)
    {
		uchar *row = outerBox.ptr(y);
        for(int x=0;x<outerBox.size().width;x++)
        {
            if(row[x]>=128)
            {
                 int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,64));
                  if(area>max)
                 {
                     maxPt = Point(x,y);
                     max = area;
                 }
            }
        }
    }
	floodFill(outerBox, maxPt, CV_RGB(255,255,255));
	for(int y=0;y<outerBox.size().height;y++)
    {
        uchar *row = outerBox.ptr(y);
        for(int x=0;x<outerBox.size().width;x++)
        {
            if(row[x]==64)
            {
                int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
	
	erode(outerBox, outerBox, kernel);
	vector<Vec2f> lines;
    HoughLines(outerBox, lines, 1, CV_PI/180, 200);
	lines = getBox(lines);
	
	return lines;
}

vector<CvPoint> corners(vector<Vec2f> lines)
{
	vector<CvPoint> points;
	CvPoint intersec;
	float m1,m2,c1,c2;
	for(int i=0;i<lines.size();i++)
	{
		for(int j=i;j<lines.size();j++)
		{
			if(lines[i][1]-lines[j][1]>0.25 || lines[j][1]-lines[i][1]>0.25)
			{
				m1=-1/tan(lines[i][1]);
				m2=-1/tan(lines[j][1]);
				c1=lines[i][0]/sin(lines[i][1]);
				c2=lines[j][0]/sin(lines[j][1]);
				intersec=cvPoint(((c2-c1)/(m1-m2)),((m1*c2-m2*c1)/(m1-m2)));
				points.push_back(intersec);
				printf("%f   %f    %f        %f\n",m1,m2,c1,c2);
			
			}

			

		}
		
	}
	for(int i=0;i<points.size();i++)
	{
		printf("%d    %d\n",points[i].x,points[i].y);
	}

	return points;




}

vector<CvPoint> sortPoints(vector<CvPoint> points,int x,int y)//returns  points of the grid according to top left bottom right
{
	vector<CvPoint> newPoints;
	int min1=points[0].x+points[0].y;
	int max1=points[0].x+points[0].y;
	int mid_up_i=abs(points[0].x - x/2 + points[0].y);
	int mid_down_i=abs(points[0].x - x/2 + points[0].y - y);
	int bottom_left_i=abs(points[0].x + points[0].y - y);
	int top_right_i=abs(points[0].x - x + points[0].y );
	CvPoint top_left=points[0];
	CvPoint right_bottom=points[0];
	CvPoint mid_up=points[0];
	CvPoint mid_down=points[0];
	CvPoint left_bottom=points[0];
	CvPoint top_right=points[0];
	for(int i=0;i<points.size();i++)
	{
		if(min1>points[i].x+points[i].y)
		{
			min1=points[i].x+points[i].y;
			top_left=points[i];
		}
		if(max1<points[i].x+points[i].y)
		{
			max1=points[i].x+points[i].y;
			right_bottom=points[i];
		}

		if(abs(points[i].x - x/2 + points[i].y)<mid_up_i)
		{
			mid_up_i=abs(points[i].x - x/2 + points[i].y);
			mid_up=points[i];
		}

		if(abs(points[i].x - x/2 + points[i].y - y)<mid_down_i)
		{
			mid_down_i=abs(points[0].x - x/2 + points[0].y - y);
			mid_down=points[i];
		}

		if(abs(points[i].x + points[i].y - y)<bottom_left_i)
		{
			bottom_left_i=abs(points[i].x + points[i].y - y);
			left_bottom=points[i];
		}

		if(abs(points[i].x - x + points[i].y)<top_right_i)
		{
			top_right_i=abs(points[i].x + points[i].y - y);
			top_right=points[i];
		}
	}
	newPoints.push_back(top_left);
	newPoints.push_back(left_bottom);
	newPoints.push_back(mid_down);
	newPoints.push_back(right_bottom);
	newPoints.push_back(top_right);
	newPoints.push_back(mid_up);
	
	return newPoints;

}

CvPoint getLastPoint(vector<CvPoint> pts,CvPoint currentPoint,CvPoint prevPoint,IplImage* start)
{
	vector<CvPoint> collosionLine;
	vector<CvPoint> shortLine;
	vector<CvPoint> secondCollision;

	int a=0;
	double prewLength =  sqrt(pow((double)(prevPoint.x-pts.at(0).x),2) + pow((double)(prevPoint.y-pts.at(0).y),2))+sqrt(pow((double)(prevPoint.x-pts.at(4).x),2) + pow((double)(prevPoint.y-pts.at(4).y),2));
	double currentLength =  sqrt(pow((double)(currentPoint.x-pts.at(0).x),2) + pow((double)(currentPoint.y-pts.at(0).y),2))+sqrt(pow((double)(currentPoint.x-pts.at(4).x),2) + pow((double)(currentPoint.y-pts.at(4).y),2));
		
	double prewLengthShort =  sqrt(pow((double)(prevPoint.x-pts.at(0).x),2) + pow((double)(prevPoint.y-pts.at(0).y),2))+sqrt(pow((double)(prevPoint.x-pts.at(1).x),2) + pow((double)(prevPoint.y-pts.at(1).y),2));
	double currentLengthShort =  sqrt(pow((double)(currentPoint.x-pts.at(0).x),2) + pow((double)(currentPoint.y-pts.at(0).y),2))+sqrt(pow((double)(currentPoint.x-pts.at(1).x),2) + pow((double)(currentPoint.y-pts.at(1).y),2));
	if(prewLength>currentLength)
			{
			
			collosionLine.push_back(pts.at(0));
			collosionLine.push_back(pts.at(4));
			secondCollision.push_back(pts.at(1));
			secondCollision.push_back(pts.at(3));

		}
		else
		{
			collosionLine.push_back(pts.at(1));
			collosionLine.push_back(pts.at(3));
			secondCollision.push_back(pts.at(0));
			secondCollision.push_back(pts.at(4));

		}

		if(prewLengthShort>currentLengthShort)
		{
			if( (((pts.at(0)).x==collosionLine.at(0).x) && ((pts.at(0)).y==collosionLine.at(0).y)) || (((pts.at(0)).x==collosionLine.at(1).x) && ((pts.at(0)).y==collosionLine.at(1).y))){
			shortLine.push_back(pts.at(0));
			shortLine.push_back(pts.at(1));
			}else{
			shortLine.push_back(pts.at(1));
			shortLine.push_back(pts.at(0));
			}

		}

		else
		{
			if( (((pts.at(3)).x==collosionLine.at(0).x) && ((pts.at(3)).y==collosionLine.at(0).y)) || (((pts.at(3)).x==collosionLine.at(1).x) && ((pts.at(3)).y==collosionLine.at(1).y))){
			shortLine.push_back(pts.at(3));
			shortLine.push_back(pts.at(4));
			}
			else
			{
				shortLine.push_back(pts.at(4));
				shortLine.push_back(pts.at(3));
			}
		}
		
		
		if( (robot==0 && ((shortLine.at(0).x==pts.at(0).x && shortLine.at(0).y==pts.at(0).y) || (shortLine.at(1).x==pts.at(0).x && shortLine.at(1).y==pts.at(0).y)) ) ||  (robot==1 && ((shortLine.at(0).x==pts.at(4).x && shortLine.at(0).y==pts.at(4).y) || (shortLine.at(1).x==pts.at(4).x && shortLine.at(1).y==pts.at(4).y)) ) )
		{
			double m1 = (double)(collosionLine.at(0).y-collosionLine.at(1).y)/(collosionLine.at(0).x-collosionLine.at(1).x);
			double m2 = (double)(prevPoint.y-currentPoint.y)/(prevPoint.x-currentPoint.x);
			double c1 = (double)(collosionLine.at(1).y-m1*collosionLine.at(1).x);
			double c2 = (double)(currentPoint.y-m2*currentPoint.x);
			double mshort=(double)(shortLine.at(0).y-shortLine.at(1).y)/(shortLine.at(0).x-shortLine.at(1).x);
			double cshort= (double)(shortLine.at(1).y-m1*shortLine.at(1).x);
			CvPoint lastPoint= cvPoint((int)((cshort-c2)/(m2-mshort)),(int)((m2*cshort-mshort*c2)/(m2-mshort)));
			
			if( sqrt(pow((double)(shortLine.at(0).x-shortLine.at(1).x),2)+pow((double)(shortLine.at(0).y-shortLine.at(1).y),2))>
				sqrt(pow((double)(shortLine.at(0).x-lastPoint.x),2)+pow((double)(shortLine.at(0).y-lastPoint.y),2)) &&  
				sqrt(pow((double)(shortLine.at(0).x-shortLine.at(1).x),2)+pow((double)(shortLine.at(0).y-shortLine.at(1).y),2))>
				sqrt(pow((double)(shortLine.at(1).x-lastPoint.x),2)+pow((double)(shortLine.at(1).y-lastPoint.y),2))
				)
			{
				cvCircle(start,lastPoint,10,cvScalar(0,0,255),1,8,0);//printf("%d    %d\n",lastPoint.x,lastPoint.y);
				return lastPoint;
				
			}
			else
			{
				CvPoint collosionPointAtLong = cvPoint((int)((c2-c1)/(m1-m2)),(int)((m1*c2-m2*c1)/(m1-m2)));
				int colX = collosionPointAtLong.x;
				int colY = collosionPointAtLong.y;
				double t = (double)(m1*colX-prevPoint.x*m1-(colY-prevPoint.y))/(1+pow(m1,2));
				double Xn = (double)(2*(colX-m1*t)-prevPoint.x);
				double Yn = (double)(2*(t+colY)-prevPoint.y);
				double m4 = (double)((Yn-colY)/(Xn-colX));
				double t2 = (double)((colY-shortLine.at(0).y)*(shortLine.at(1).x-shortLine.at(0).x)-(colX-shortLine.at(0).x)*(shortLine.at(1).y-shortLine.at(0).y))/(shortLine.at(1).y-shortLine.at(0).y-m2*(shortLine.at(1).x-shortLine.at(0).x));
				lastPoint= cvPoint((int)(colX+t2),(int)(m4*t2+colY));
				CvPoint collosionPointAtLong2=cvPoint(0,0);
				while(sqrt(pow((double)(shortLine.at(0).x-shortLine.at(1).x),2)+pow((double)(shortLine.at(0).y-shortLine.at(1).y),2))<sqrt(pow((double)(shortLine.at(0).x-lastPoint.x),2)+pow((double)(shortLine.at(0).y-lastPoint.y),2)) && a<3)
				{	
					double m3 = (double)(secondCollision.at(0).y-secondCollision.at(1).y)/(secondCollision.at(0).x-secondCollision.at(1).x);
					double c3 = (double)(secondCollision.at(1).y-m3*secondCollision.at(1).x);
					double c4 = (double)(Yn-m4*Xn);
					collosionPointAtLong2 = cvPoint((int)((c3-c4)/(m4-m3)),(int)((m4*c3-m3*c4)/(m4-m3)));
		
					int colX2 = collosionPointAtLong2.x;
					int colY2= collosionPointAtLong2.y;
					double k= (double)(m3*colX2-Xn*m3-(colY2-Yn))/(1+pow(m3,2));
					double Xn2 = (double)(2*(colX2-m3*k)-Xn);
					double Yn2 =  (double)(2*(k+colY2)-Yn);
					m4 =(double)((Yn2-colY2)/(Xn2-colX2));
					double k2 = (double)((colY2-shortLine.at(0).y)*(shortLine.at(1).x-shortLine.at(0).x)-(colX2-shortLine.at(0).x)*(shortLine.at(1).y-shortLine.at(0).y))/(shortLine.at(1).y-shortLine.at(0).y-m4*(shortLine.at(1).x-shortLine.at(0).x));
					lastPoint=cvPoint((int)(colX2+k2),(int)(m4*k2+colY2));
					CvPoint temp=shortLine.at(0);
					shortLine.at(0)=shortLine.at(1);
					shortLine.at(1)=temp;
					vector<CvPoint> tempLine=collosionLine;
					collosionLine=secondCollision;
					secondCollision=tempLine;
					a++;

				}cvCircle(start,lastPoint,10,cvScalar(0,0,255),1,8,0);
			return lastPoint;
			}
		}
		else return cvPoint(-1,-1);
}

boolean checkInSameSide(vector<CvPoint> line,vector<CvPoint> toCheck)
{
	double m=(double)(line.at(0).y-line.at(1).y)/(line.at(0).x-line.at(1).x);
	double c=(double)(line.at(0).y*line.at(1).x - line.at(1).y*line.at(0).x)/(line.at(1).x-line.at(0).x);
	if((m*toCheck.at(0).x + c - toCheck.at(0).y)*(m*toCheck.at(1).x + c - toCheck.at(1).y)>0 )
		return true;
	else
		return false;
}

vector<CvPoint> DetectRight(CvPoint head,CvPoint pts1,CvPoint pts2)
{
	vector<CvPoint> inOrder;
	if(head.y>pts1.y && head.y>pts2.y)
	{
		if(pts1.x>pts2.x)
		{
			inOrder.push_back(pts1);
			inOrder.push_back(pts2);
		}
		else
		{
			inOrder.push_back(pts2);
			inOrder.push_back(pts1);
		}
	}

	else if(head.y<pts1.y && head.y<pts2.y)
	{
		if(pts1.x>pts2.x)
		{
			inOrder.push_back(pts2);
			inOrder.push_back(pts1);
		}
		else
		{
			inOrder.push_back(pts1);
			inOrder.push_back(pts2);
		}
	}

	else
	{
		if(head.x>pts1.x && head.x>pts2.x)
		{
			if(pts1.y>pts2.y)
			{
				inOrder.push_back(pts2);
				inOrder.push_back(pts1);
			}
			else
			{
				inOrder.push_back(pts1);
				inOrder.push_back(pts2);
			}
		}

		else if(head.x<pts1.x && head.x<pts2.x)
		{
			if(pts1.y>pts2.y)
			{
				inOrder.push_back(pts1);
				inOrder.push_back(pts2);
			}
			else
			{
				inOrder.push_back(pts2);
				inOrder.push_back(pts1);
			}
		}
	}
	return inOrder;
	//inOrder.at(0) --> right vertex		inOrder.at(1) --> left vertex
}

boolean findWayToRotate(CvPoint Ball,CvPoint leftV,CvPoint rightV)
{
	if(getLength(Ball,leftV)>getLength(Ball,rightV))
		return true;	//turn right,clockwise
	else
		return false;	//turn left, anty clockwise
}

int main()
{
	boolean direction=findWayToRotate(cvPoint(10,10),cvPoint(1,1),cvPoint(5,1));
	if(direction)
		printf("turn right");
	else
		printf("turn left");
	return 0;
}