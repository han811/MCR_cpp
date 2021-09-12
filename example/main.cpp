/*
This file is part of LMPL.

    LMPL is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LMPL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the Lesser
    GNU General Public License for more details.

    You should have received a copy of the Lesser GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "Geometric2DCSpace.h"
#include "PlannerTest.h"
#include <misc/Random.h>
// #include "MyExplicitCSpace.h"
#include <MotionPlanning/MyExplicitCSpace.h>
#include <MotionPlanning/ExplainingPlanner.h>
#include "misc/Miscellany.h"
#include <iostream>
#include "Timer.h"
#include <random>
#include <fstream>
#include <time.h>
using namespace std;

#define EUCLIDEAN_SPACE 1

void SetupBoxObstacle(Geometric2DCSpace& cspace,double width,double height)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);
	AABB2D temp;
	temp.bmin.set(0.5-0.5*width,0.5-0.5*height);
	temp.bmax.set(0.5+0.5*width,0.5+0.5*height);
	cspace.Add(temp);
}

void SetupTrianglePassage(Geometric2DCSpace& cspace,double passageWidth,double baseWidth)
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

void SetupCircularField(Geometric2DCSpace& cspace,int numCircles,double passageWidth)
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

void SetupKink(Geometric2DCSpace& cspace,double passageWidth,double width,double kinkLength)
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

void SetupWindy(Geometric2DCSpace& cspace,int numWinds,double passageWidth,double kinkWidth)
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

void SetupPassage(Geometric2DCSpace& cspace,double passageWidth,double width)
{
	AABB2D temp;
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(1,1);

	temp.bmin.set(0.5-0.5*width,0);
	temp.bmax.set(0.5+0.5*width,0.5-passageWidth);
	cspace.Add(temp);
	temp.bmin.set(0.5-0.5*width,0.5);
	temp.bmax.set(0.5+0.5*width,1);
	cspace.Add(temp);
}








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



void MCRsetup(MyExplicitCSpace& cspace, double width, double height, double radius, int num)
{
	cspace.euclideanSpace = EUCLIDEAN_SPACE;
	cspace.domain.bmin.set(0,0);
	cspace.domain.bmax.set(width,height);
	
	
	AABB2D temp;
	// static obstacle 0
	temp.bmin.set(0.3*width,0.1*height);
	temp.bmax.set(0.7*width,0.4*height);
	cspace.Add(temp);
	// static obstacle 1
	temp.bmin.set(0.3*width,0.6*height);
	temp.bmax.set(0.7*width,0.9*height);
	cspace.Add(temp);



	// // static obstacle 0
	// temp.bmin.set(3.0,1.0);
	// temp.bmax.set(7.0,4.0);
	// cspace.Add(temp);
	// // static obstacle 1
	// temp.bmin.set(3.0,6.0);
	// temp.bmax.set(7.0,9.0);
	// cspace.Add(temp);
	
	Circle2D temp2;
	temp2.radius = radius;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	for(int i=0; i<num; i++){
		double ratio1 = dis(gen);
		double ratio2 = dis(gen);
		temp2.center.x = width*ratio1;
		temp2.center.y = height*ratio2;
		cspace.Add(temp2);
		// temp2.center.x = width*0.5;
		// temp2.center.y = height*0.5;
		// cspace.Add(temp2);
		// temp2.center.x = width*0.25;
		// temp2.center.y = height*0.25;
		// cspace.Add(temp2);
		// temp2.center.x = width*0.75;
		// temp2.center.y = height*0.75;
		// cspace.Add(temp2);
	}
}


int main(int argc,char** argv)
{
	RandHelper::srand(time(NULL));
	// Geometric2DCSpace cspace;
	// SetupPassage(cspace,0.01,0.3);

	// MotionPlannerFactory factory;
	// factory.type = MotionPlannerFactory::PRM;
	// // factory.type = MotionPlannerFactory::SBL;

	// Config start(2),goal(2);
	// start[0] = 0.1;
	// start[1] = 0.2;
	// goal[0] = 0.9;
	// goal[1] = 0.2;
	// int numIters=10000;
	// PrintPlannerTest(factory,&cspace,start,goal,10,
	// 		 numIters);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	double width, height;
	width = 5.0;
	height = 5.0;

	int total_try = 10;
	for(int data_count=0; data_count<total_try; data_count++){
		// /* Set up space here */
		// Geometric2DCSpace myspace;
		MyExplicitCSpace myspace;
		// SetupPassage(myspace,0.0,0.3);
		// MCRsetup(myspace,10.0,10.0,7.5,25);
		MCRsetup(myspace,width,height,3.5,10);

		/* Set up planner and set parameters (default values shown here) */
		ErrorExplainingPlanner planner(&myspace);
		planner.numConnections = 10;        //compute k-connected PRM
		planner.connectThreshold = ConstantHelper::Inf;     //haven't tested this setting much
		planner.expandDistance = 0.1;       //how far to expand the PRM toward a random configuration at each iteration
		planner.goalConnectThreshold = 0.5; //distance at which the planner attempts to connect configurations directly to the goal
		planner.usePathCover = true;        //keep this to true, otherwise performance can be quite bad
		// planner.updatePathsComplete = false;//governs whether greedy or complete explanation set updates are used.  Can play with this.
		planner.updatePathsComplete = true;//governs whether greedy or complete explanation set updates are used.  Can play with this.
		/* Set up planner */

		planner.obstacleWeights = vector<double>(myspace.NumObstacles(),1);
		planner.obstacleWeights[0] = ConstantHelper::Inf;
		planner.obstacleWeights[1] = ConstantHelper::Inf;

		Config start(2),goal(2);
		bool sig = true;
		while(sig){
			sig = false;
			start[0] = dis(gen)*width;
			start[1] = dis(gen)*height;
			goal[0] = dis(gen)*width;
			goal[1] = dis(gen)*height;
			Point2D temp_start(start[0],start[1]), temp_goal(goal[0],goal[1]);

			for(int i=0; i<planner.space->aabbs.size(); i++){
				if(planner.space->aabbs[i].contains(temp_start)){
					sig=true;
				}
				if(planner.space->aabbs[i].contains(temp_goal)){
					sig=true;
				}
			}
		}
		start[0] = 0.1;
		start[1] = 0.2;
		goal[0] = 4.9;
		goal[1] = 4.9;



		planner.Init(start,goal);

		/* Set up an explanation limit expansion schedule, up to 5000 iterations */
		vector<int> schedule(5);
		schedule[0] = 1000;
		schedule[1] = 2000;
		schedule[2] = 3000;
		schedule[3] = 4000;
		schedule[4] = 5000;
		// schedule[5] = 60000;
		// schedule[6] = 70000;
		// schedule[7] = 80000;
		// schedule[8] = 90000;
		// schedule[9] = 100000;
		
		/* Start planning */
		vector<int> path;
		Subset cover;
		Timer timer;



		planner.Plan(0,schedule,path,cover);
		// cout << start[0] << " " << start[1] << '\n';
		// cout << goal[0] << " " << goal[1] << '\n';
		// while(path.size()==2){
		// 	cout << "plan one more time" << '\n';
		// 	planner.Plan(0,schedule,path,cover);
		// }

		cout << "path size: " << path.size() << '\n';
		
		for(int i=0; i<path.size(); i++)
			cout << path[i] << " " << planner.roadmap.nodes[path[i]].q[0] << " " << planner.roadmap.nodes[path[i]].q[1] << '\n';
		cout << "plan time: " << timer.ElapsedTime() << '\n';
		//simple print (integers):
		cout<<"Best cover: "<<cover<<endl;

		//or pretty print (obstacle names):
		bool sig2 = true;
		cout<<"Best cover:"<<endl;
		for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++){
			cout<<"  "<<myspace.ObstacleName(*i)<<endl;
			if(myspace.ObstacleName(*i)==string("Obs[0]") || myspace.ObstacleName(*i)==string("Obs[1]")){
				sig2 = false;
			}
		}
		
		if(sig2){
			ofstream fout;
			string s;
			s = "data/data";
			s += to_string(data_count);
			s += ".txt";
			fout.open(s.c_str());
			fout << "Nodes" << '\n';
			for(int i=0; i<planner.roadmap.nodes.size(); i++){
				fout << planner.roadmap.nodes[i].q[0] << " " << planner.roadmap.nodes[i].q[1] << '\n';
			}
			fout << "Edges" << '\n';
			for(int i=0; i<planner.roadmap.nodes.size(); i++){
				fout << i << " : ";
				for(int j=0; j<planner.roadmap.nodes.size(); j++){
					if(i!=j){
						if(planner.roadmap.HasEdge(i,j)){
							fout << j << " ";
						}
					}
				}
				fout << '\n';
			}
			fout << "Path" << '\n';
			for(int i=0; i<path.size(); i++){
				fout << path[i] << " ";
			}
			fout << '\n';
			fout << "Cover" << '\n';
			for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++)
				fout << myspace.ObstacleName(*i) << " ";
			fout << '\n';
			fout << "Obstacles" << '\n';
			fout << "aabbs" << '\n';
			for(int i=0; i<planner.space->aabbs.size(); i++){
				fout << i << '\n';
				fout << planner.space->aabbs[i].bmin[0] << " " << planner.space->aabbs[i].bmin[1] << '\n';
				fout << planner.space->aabbs[i].bmax[0] << " " << planner.space->aabbs[i].bmax[1] << '\n';
			}
			fout << "circles" << '\n';
			fout << planner.space->circles[0].radius << '\n';
			for(int i=0; i<planner.space->circles.size(); i++){
				fout << planner.space->circles[i].center[0] << " " << planner.space->circles[i].center[1] << '\n';
			}
			fout << "Time" << '\n';
			fout << timer.ElapsedTime() << '\n';
		}
	}

	return 0;
}


