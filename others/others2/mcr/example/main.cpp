#include <misc/Random.h>
#include <MotionPlanning/MyExplicitCSpace.h>
#include <MotionPlanning/ExplainingPlanner.h>
#include "misc/Miscellany.h"
#include <iostream>
#include "Timer.h"
#include <random>
#include <fstream>
#include <time.h>
#include <env.h>
using namespace std;

int main(int argc,char** argv)
{
	RandHelper::srand(time(NULL));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	double width, height;
	width = 12.0;
	height = 12.0;
	// height = 8.0;

	int total_try = 2;
	for(int data_count=0; data_count<total_try; data_count++){
		try{

			/* Set up space here */
			MyExplicitCSpace myspace;
			vector<int> sectors = MCRsetup(myspace,width,height,height/12.0+0.05,12);

			/* Set up planner and set parameters (default values shown here) */
			ErrorExplainingPlanner planner(&myspace);
			planner.numConnections = 10;        //compute k-connected PRM
			planner.connectThreshold = ConstantHelper::Inf;     //haven't tested this setting much
			planner.expandDistance = 0.1;       //how far to expand the PRM toward a random configuration at each iteration
			planner.goalConnectThreshold = 0.5; //distance at which the planner attempts to connect configurations directly to the goal
			planner.usePathCover = true;        //keep this to true, otherwise performance can be quite bad
			planner.updatePathsComplete = true;//governs whether greedy or complete explanation set updates are used.  Can play with this.
			
			/* Set up planner */
			planner.obstacleWeights = vector<double>(myspace.NumObstacles(),1);
			planner.obstacleWeights[0] = ConstantHelper::Inf;
			planner.obstacleWeights[1] = 0;
			planner.obstacleWeights[5] = 0;
			planner.obstacleWeights[6] = 0;
			planner.obstacleWeights[9] = 0;
			planner.obstacleWeights[12] = 0;
			// planner.obstacleWeights[1] = ConstantHelper::Inf;

			Config start(2),goal(2);
			bool sig = true;
			while(sig){
				sig = true;
				start[0] = dis(gen)*width;
				start[1] = dis(gen)*height;
				goal[0] = dis(gen)*width;
				goal[1] = dis(gen)*height;
				// Point2D temp_start(start[0],start[1]), temp_goal(goal[0],goal[1]);

				// for(int i=0; i<planner.space->aabbs.size(); i++){
				// 	if(planner.space->aabbs[i].contains(temp_start)){
				// 		sig=true;
				// 	}
				// 	if(planner.space->aabbs[i].contains(temp_goal)){
				// 		sig=true;
				// 	}
				// }
				if(0.1<=start[0] && start[0]<=(width/6.0) && 0.1<=start[1] && start[1]<=(height/6.0) && goal[0]>=(width*5.0/6.0) && (width-0.1)>goal[0] && (height-0.1)>goal[1] && goal[1]>=(height*5.0/6.0)){
					sig=false;
				}
			}
			// start[0] = 0.1;
			// start[1] = 0.2;
			// goal[0] = 4.9;
			// goal[1] = 4.9;



			planner.Init(start,goal);

			/* Set up an explanation limit expansion schedule, up to 5000 iterations */
			vector<int> schedule(5);
			// vector<int> schedule(10);
			schedule[0] = 2000;
			schedule[1] = 4000;
			schedule[2] = 6000;
			schedule[3] = 8000;
			schedule[4] = 10000;
			// schedule[5] = 12000;
			// schedule[6] = 14000;
			// schedule[7] = 16000;
			// schedule[8] = 18000;
			// schedule[9] = 20000;
			// schedule[10] = 22000;
			// schedule[11] = 24000;
			// schedule[12] = 26000;
			// schedule[13] = 28000;
			// schedule[14] = 30000;
			// schedule[15] = 32000;
			// schedule[16] = 34000;
			// schedule[17] = 36000;
			// schedule[18] = 38000;
			// schedule[19] = 40000;
			// schedule[0] = 1000*2;
			// schedule[1] = 2000*2;
			// schedule[2] = 3000*2;
			// schedule[3] = 4000*2;
			// schedule[4] = 5000*2;
			// schedule[5] = 6000*2;
			// schedule[6] = 7000*2;
			// schedule[7] = 8000*2;
			// schedule[8] = 9000*2;
			// schedule[9] = 10000*2;
			
			/* Start planning */
			vector<int> path;
			Subset cover;
			Timer timer;


			// planner.Plan(0,schedule,path,cover);
			vector<int> v(1);
			planner.algorithm1(0,schedule,path,cover,v);
			double plan_time = timer.ElapsedTime();
			cout << "path size: " << path.size() << '\n';
			for(int i=0; i<path.size(); i++)
				cout << path[i] << " " << planner.roadmap.nodes[path[i]].q[0] << " " << planner.roadmap.nodes[path[i]].q[1] << '\n';
			cout << "plan time: " << plan_time << '\n';
			
			cout<<"Best cover: "<<cover<<endl;
			bool sig2 = true;
			for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++){
				if(myspace.ObstacleName(*i)==string("Obs[0]") || myspace.ObstacleName(*i)==string("Obs[1]")){
					sig2 = false;
				}
			}
			cout << '\n';
			
			if(sig2){
				ofstream fout;
				string s;
				s = "data/data_cpp/data";
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
				fout << plan_time << '\n';
				fout << "sectors" << '\n';
				fout << sectors[0] << '\n';
				fout << sectors[1] << '\n';
				fout << sectors[2] << '\n';
				fout << sectors[3] << '\n';
			}
		}
		catch(std::exception& e){
			cout << "Exception caught: " << e.what() << '\n';
		}
		cout << "data num: " << data_count << '\n';
	}

	return 0;
}


