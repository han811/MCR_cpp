#include <MotionPlanning/MyExplicitCSpace.h>
#include <MotionPlanning/ExplainingPlanner.h>
#include <env.h>
#include <fstream>
#include <iostream>
#include <vector>

void InitializedPlanner(ErrorExplainingPlanner &planner)
{
    planner.numConnections = 20;        //compute k-connected PRM
    planner.connectThreshold = ConstantHelper::Inf;     //haven't tested this setting much
    planner.expandDistance = 0.5;       //how far to expand the PRM toward a random configuration at each iteration
    planner.goalConnectThreshold = 0.5; //distance at which the planner attempts to connect configurations directly to the goal
    planner.usePathCover = true;        //keep this to true, otherwise performance can be quite bad
    planner.updatePathsComplete = true;//governs whether greedy or complete explanation set updates are used.  Can play with this.
}

void SetupObstacleWeights(ErrorExplainingPlanner &planner, MyExplicitCSpace &myspace, vector<double> obstacle_weights)
{
    planner.obstacleWeights = obstacle_weights;
    planner.obstacleWeights[0] = ConstantHelper::Inf;
    planner.obstacleWeights[1] = ConstantHelper::Inf;
}

void SaveResult(ErrorExplainingPlanner &planner,MyExplicitCSpace &myspace, vector<int> &path, Subset &cover, int data_count, double plan_time, vector<int> sectors, int iteration)
{
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
    fout << "iterations" << '\n';
    fout << iteration << '\n';
}

void SaveObstaclesResult(ErrorExplainingPlanner &planner, MyExplicitCSpace &myspace, Subset &cover, int data_count)
{
    ofstream fout;
    string s;
    s = "data/obstacles/";
    s += to_string(data_count);
    s += ".txt";
    fout.open(s.c_str());
    for(int i=0; i<planner.space->circles.size(); i++){
        fout << planner.space->circles[i].center[0] << " " << planner.space->circles[i].center[1] << '\n';
    }

    fout << "Cover" << '\n';
    for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++){
        if(i!=(--cover.items.end())){
            fout << *i << " ";
        }
        else{
            fout << *i;
        }
    }
}