#include <misc/Random.h>
#include <MotionPlanning/MyExplicitCSpace.h>
#include <MotionPlanning/ExplainingPlanner.h>
#include "misc/Miscellany.h"
#include <iostream>
#include "Timer.h"
#include <random>
#include <fstream>
#include <time.h>
#include <Utils.h>

#include "socket/ClientSocket.h"
#include "socket/SocketException.h"

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

	int total_try = 10;

	// ClientSocket client_socket ( "localhost", 8080 );
	for(int data_count=0; data_count<total_try; data_count++){
		cout << "data num: " << data_count << '\n';
		/* Set up space here */
		MyExplicitCSpace myspace;
		vector<int> sectors = MCRsetup(myspace,width,height,height/12.0+0.05,50);

		/* Set up planner and set parameters (default values shown here) */
		ErrorExplainingPlanner planner(&myspace);
		InitializedPlanner(planner);
		
		/* Set up planner */
		vector<double> obstacle_weights(myspace.NumObstacles(),1.0);
		// obstacle_weights[0] = ConstantHelper::Inf;
		// obstacle_weights[1] = ConstantHelper::Inf;
		SetupObstacleWeights(planner,myspace,obstacle_weights);

		/* Check is_static and labels from GNN */
		vector<bool> is_static(myspace.NumObstacles(),false);
		is_static[0] = true;
		is_static[1] = true;
		vector<bool> labels(myspace.NumObstacles(),false);
		// labels[2] = true;
		// labels[5] = true;
		// labels[24] = true;


		/* make start and goal configurations */
		Config start(2),goal(2);
		bool sig = true;
		while(sig){
			sig = true;
			start[0] = dis(gen)*width;
			start[1] = dis(gen)*height;
			goal[0] = dis(gen)*width;
			goal[1] = dis(gen)*height;
			if(0.1<=start[0] && start[0]<=(width/6.0) && 0.1<=start[1] && start[1]<=(height/6.0) && goal[0]>=(width*5.0/6.0) && (width-0.1)>goal[0] && (height-0.1)>goal[1] && goal[1]>=(height*5.0/6.0)){
				sig=false;
			}
		}
		planner.Init(start,goal,is_static,labels);

		/* Start planning */
		vector<int> path;
		Subset cover;
		
		/* Set up an explanation limit expansion schedule, up to 5000 iterations */
		vector<int> schedule(5);
		schedule[0] = 2000;
		schedule[1] = 4000;
		schedule[2] = 6000;
		schedule[3] = 8000;
		schedule[4] = 10000;

		// for(int gnn=0; gnn<5; gnn++){
		// 	std::string reply;
		// 	string obstacles_string;
		// 	for(int ob_idx=0; ob_idx<myspace.circles.size(); ob_idx++){
		// 		obstacles_string += to_string(myspace.circles[ob_idx].center.x);
		// 		obstacles_string += " ";
		// 		obstacles_string += to_string(myspace.circles[ob_idx].center.y);
		// 	}
		// 	client_socket << obstacles_string;
		// 	client_socket >> reply;
		// 	planner.Plan(0,schedule,path,cover);
		// }

		Timer timer;
		planner.Plan(0,schedule,path,cover);
		double plan_time = timer.ElapsedTime();

		cout << "plan time: " << plan_time << '\n';
		for(int i=0; i<planner.progress_times.size(); i++){
			cout << "planning time " << i << " : " << planner.progress_times[i] << '\n';
		}
		
		cout<<"Best cover: "<<cover<<endl;
		bool sig2 = true;
		for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++){
			if(myspace.ObstacleName(*i)==string("Obs[0]") || myspace.ObstacleName(*i)==string("Obs[1]")){
				sig2 = false;
			}
		}
		cout << '\n';
		if(sig2){
			SaveResult(planner, myspace, path, cover, data_count, plan_time, sectors);
		}
	}
	return 0;
}


