#ifndef OBSTACLE_H
#define OBSTACLE_H

#include <vector>
#include <string>

using namespace std;

bool line_intersection(pair<pair<double,double>,pair<double,double>> line1, pair<pair<double,double>,pair<double,double>> line2);
double line_point_distance(pair<double,double> p1, pair<double,double> p2, pair<double,double> c);
bool rectangle_circle_center(pair<double,double> lb, pair<double,double> rt, pair<double,double> c);
bool rectangle_intersect_circle(pair<double,double> c, double radius, vector<pair<double,double>> rect);


class Obstacle
{
  public:
    Obstacle(){};
    Obstacle(pair<double,double> init_position, string obstacle_type, double radius, vector<pair<double,double>> polygon_points, bool is_static);
    string obstacle_type;
    double x,y;
    double radius;
    vector<pair<double,double>> polygon;
    bool is_static;

    bool feasibility_check(Obstacle target_object);
    bool feasibility_check_point(pair<double,double> point);
};

class Obstacles
{
  public:
    Obstacles(){};
    Obstacles(int num, double obs_radius, vector<Obstacle> static_obstacles, double mwidth, double mheight, string movable_obstacle_type, string static_obstacle_type);

    double radius;
    vector<Obstacle> static_obstacles;
    vector<Obstacle> movable_obstacles;
    int num=0;
    int movable_num=0;
    int static_num=0;
    double mwidth = mwidth;
    double mheight = mheight;
    string movable_obstacle_type = movable_obstacle_type;
    string static_obstacle_type = static_obstacle_type;

    bool feasibility_check(Obstacle object);
    bool feasibility_check_point(pair<double,double> point);
};

#endif