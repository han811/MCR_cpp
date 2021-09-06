#ifndef GRAPH
#define GRAPH

#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>
#include "Obstacle.h"

using namespace std;

class Graph{
  public:
    Graph(int q_s, pair<double,double> q_s_val, double mwidth, double mheight, int ob_num, double ob_radius, vector<Obstacle> static_obstacles);
    Graph(int q_s, pair<double,double> q_s_val, int q_g, pair<double,double> q_g_val, double mwidth, double mheight, int ob_num, double ob_radius, vector<Obstacle> static_obstacles);
    
    unordered_map<int,pair<double,double>> nodes;
    unordered_map<int,vector<int>> edges;

    int node_num = 0;

    int start_node = 0;
    int goal_node = 1;

    Obstacles obstacles;    

    double mwidth;
    double mheight;
    int ob_num;
    double ob_radius;

    int k=0;

  // Expand-Roadmap
    pair<double,double> Sample();
    int Closest(pair<double,double> q_d);

    void Cleanup();

};







#endif