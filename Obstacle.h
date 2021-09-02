#ifndef OBSTACLE_H
#define OBSTACLE_H

#include <vector>
#include <string>

using namespace std;

class Obstacle
{
  public:
    Obstacle(vector<pair<int,int>> init_position, string obstacle_type, double radius, vector<pair<int,int>> polygon_points, bool is_static);
    string obstacle_type = obstacle_type;

};

#endif