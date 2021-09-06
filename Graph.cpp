#include "Graph.h"
#include "Obstacle.h"
#include <random>

Graph::Graph(int q_s, pair<double,double> q_s_val, double mwidth, double mheight, int ob_num, double ob_radius, vector<Obstacle> static_obstacles)
{
    nodes[q_s] = q_s_val;
    node_num+=1;
    mwidth = mwidth;
    mheight = mheight;
    ob_num = ob_num;
    ob_radius = ob_radius;
    obstacles = Obstacles(ob_num, ob_radius, static_obstacles, mwidth, mheight, "circle", "rectangle");

    k = obstacles.movable_num;
}

Graph::Graph(int q_s, pair<double,double> q_s_val, int q_g, pair<double,double> q_g_val, double mwidth, double mheight, int ob_num, double ob_radius, vector<Obstacle> static_obstacles)
{
    nodes[q_s] = q_s_val;
    nodes[q_g] = q_g_val;

    vector<int> tmp1;
    tmp1.push_back(q_g);
    edges[q_s] = tmp1;

    vector<int> tmp2;
    tmp1.push_back(q_s);
    edges[q_g] = tmp2;
    
    node_num+=1;
    mwidth = mwidth;
    mheight = mheight;
    ob_num = ob_num;
    ob_radius = ob_radius;
    obstacles = Obstacles(ob_num, ob_radius, static_obstacles, mwidth, mheight, "circle", "rectangle");

    k = obstacles.movable_num;
}

void Graph::Cleanup()
{
    nodes.clear();
    edges.clear();
}

pair<double,double> Graph::Sample()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-0.5,0.5);
    double tmp_x = dis(gen)*mwidth;
    double tmp_y = dis(gen)*mheight;

    return make_pair(tmp_x, tmp_y);
}




