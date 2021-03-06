#include <misc/Random.h>
#include <MotionPlanning/MyExplicitCSpace.h>
#include <MotionPlanning/ExplainingPlanner.h>
#include "misc/Miscellany.h"
#include <iostream>
#include <random>
#include <fstream>
#include <time.h>
using namespace std;

#define EUCLIDEAN_SPACE 1


// env setup functions 
void SetupBoxObstacle(MyExplicitCSpace& cspace,double width,double height)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);
	AABB2D temp;
	temp.bmin.set(0.5-0.5*width,0.5-0.5*height);
	temp.bmax.set(0.5+0.5*width,0.5+0.5*height);
	cspace.Add(temp);
}

void SetupTrianglePassage(MyExplicitCSpace& cspace,double passageWidth,double baseWidth)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	Triangle2D t;
	t.a.set(0.5+baseWidth*0.5,0.0);
	t.b.set(0.5,0.5-passageWidth*0.5);
	t.c.set(0.5-baseWidth*0.5,0.0);
	cspace.Add(t);
	t.a.set(0.5-baseWidth*0.5,1);
	t.b.set(0.5,0.5+passageWidth*0.5);
	t.c.set(0.5+baseWidth*0.5,1);
	cspace.Add(t);
}

void SetupCircularField(MyExplicitCSpace& cspace,int numCircles,double passageWidth)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	Circle2D c;
	double width = (1.0-2.0*passageWidth)/numCircles;
	double start = width*0.5+passageWidth;
	c.radius = passageWidth*width;
	for(int i=0;i<numCircles;i++) {
		if(i%2==0) {
			for(int j=0;j+1<numCircles;j++) {
				c.center.set(width*(double(j)+0.5)+start,width*double(i)+start);
				cspace.Add(c);
			}
		}
		else {
			for(int j=0;j<numCircles;j++) {
				c.center.set(width*double(j)+start,width*double(i)+start);
				cspace.Add(c);
			}
		}
	}
}

void SetupKink(MyExplicitCSpace& cspace,double passageWidth,double width,double kinkLength)
{
	AABB2D temp;
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	//bottom left of kink
	temp.bmin.set(0.5-0.5*width,0);
	temp.bmax.set(0.5-0.5*passageWidth,0.5-0.5*passageWidth+kinkLength*0.5);
	cspace.Add(temp);
	//bottom right of kink
	temp.bmin.set(0.5-0.5*passageWidth,0);
	temp.bmax.set(0.5+0.5*width,0.5-0.5*passageWidth-kinkLength*0.5);
	cspace.Add(temp);
	//top left of kink
	temp.bmin.set(0.5-0.5*width,0.5+0.5*passageWidth+kinkLength*0.5);
	temp.bmax.set(0.5+0.5*passageWidth,1);
	cspace.Add(temp);
	//top right of kink
	temp.bmin.set(0.5+0.5*passageWidth,0.5+0.5*passageWidth-kinkLength*0.5);
	temp.bmax.set(0.5+0.5*width,1);
	cspace.Add(temp);
}

void SetupWindy(MyExplicitCSpace& cspace,int numWinds,double passageWidth,double kinkWidth)
{
	AABB2D temp;
	//TODO: what might the 'border' be? Parameter or constant?
	double border = 0.1;

	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	double width = 1.0/double(numWinds+1);
	for(int i=0;i<numWinds;i++) {
		double center = double(i+1)/double(numWinds+1);
		if(i%2 == 1) {
			temp.bmin.set(center-width*kinkWidth,0);
			temp.bmax.set(center+width*kinkWidth,1.0-border-passageWidth);
		}
		else {
			temp.bmin.set(center-width*kinkWidth,border+passageWidth);
			temp.bmax.set(center+width*kinkWidth,1.0);
		}
		cspace.Add(temp);
	}
}

void SetupPassage(MyExplicitCSpace& cspace,double passageWidth,double width)
{
	AABB2D temp;
	Circle2D temp2;
	temp2.radius = passageWidth;
	temp2.center.x = 0.5;
	temp2.center.y = 0.5;
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	cspace.Add(temp2);
	cspace.Add(temp2);
	cspace.Add(temp2);
	cspace.Add(temp2);
	cspace.Add(temp2);
	cspace.Add(temp2);
	
	// for(int its=0; its<1000; its++){
		temp.bmin.set(0.5-0.5*width,0);
		temp.bmax.set(0.5+0.5*width,0.5-passageWidth);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0.5+passageWidth);
		temp.bmax.set(0.5+0.5*width,1);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0);
		temp.bmax.set(0.5+0.5*width,0.5-passageWidth);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0.5+passageWidth);
		temp.bmax.set(0.5+0.5*width,1);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0);
		temp.bmax.set(0.5+0.5*width,0.5-passageWidth);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0.5+passageWidth);
		temp.bmax.set(0.5+0.5*width,1);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0);
		temp.bmax.set(0.5+0.5*width,0.5-passageWidth);
		cspace.Add(temp);
		temp.bmin.set(0.5-0.5*width,0.5+passageWidth);
		temp.bmax.set(0.5+0.5*width,1);
		cspace.Add(temp);
	// }
}

vector<int> MCRsetup(MyExplicitCSpace& cspace, double width, double height, double radius, int num)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(width,height);
	
	// AABB2D temp;
	// // static obstacle 0
	// temp.bmin.set(width/6,height/6);
	// temp.bmax.set(width*5/6,height*5/12);
	// cspace.Add(temp);
	// // // static obstacle 1
	// temp.bmin.set(width/6,height*7/12);
	// temp.bmax.set(width*5/6,height*10/12);
	// cspace.Add(temp);

	Circle2D temp2;
	temp2.radius = radius;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	
	vector<int> sectors(3);
	for(int i=0; i<num; i++){
		double ratio1 = dis(gen);
		double ratio2 = dis(gen);
		temp2.center.x = width*ratio1;
		temp2.center.y = height*ratio2;
		cspace.Add(temp2);
	}

	// for(int i=0; i<num; i++){
	// 	double dice = dis(gen);
	// 	if(dice<0.33){
	// 		while(true){
	// 			double ratio1 = dis(gen);
	// 			double ratio2 = dis(gen);
	// 			if((1.0/6)<=ratio1 && ratio1<(5.0/6) && (5.0/6)<=ratio2 && ratio2<1.0){
	// 				temp2.center.x = width*ratio1;
	// 				temp2.center.y = height*ratio2;
	// 				cspace.Add(temp2);
	// 				break;		
	// 			}
	// 		}
	// 		sectors[2] += 1;
	// 	}
	// 	else if(dice<0.66){
	// 		while(true){
	// 			double ratio1 = dis(gen);
	// 			double ratio2 = dis(gen);
	// 			if((1.0/6)<=ratio1 && ratio1<(5.0/6) && (5.0/12)<=ratio2 && ratio2<(7.0/12)){
	// 				temp2.center.x = width*ratio1;
	// 				temp2.center.y = height*ratio2;
	// 				cspace.Add(temp2);
	// 				break;		
	// 			}
	// 		}
	// 		sectors[1] += 1;
	// 	}
	// 	else{
	// 		while(true){
	// 			double ratio1 = dis(gen);
	// 			double ratio2 = dis(gen);
	// 			if((1.0/6)<=ratio1 && ratio1<(5.0/6) && 0.0<=ratio2 && ratio2<(1.0/6)){
	// 				temp2.center.x = width*ratio1;
	// 				temp2.center.y = height*ratio2;
	// 				cspace.Add(temp2);
	// 				break;		
	// 			}
	// 		}
	// 		sectors[0] += 1;
	// 	}
	// }
	return sectors;
}
