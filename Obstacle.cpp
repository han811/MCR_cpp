#include "Obstacle.h"
#include <cmath>
#include <random>

bool line_intersection(pair<pair<double,double>,pair<double,double>> line1, pair<pair<double,double>,pair<double,double>> line2)
{
    double x1 = line1.first.first;
    double y1 = line1.first.second;
    double x2 = line1.second.first;
    double y2 = line1.second.second;
    double x3 = line2.first.first;
    double y3 = line2.first.second;
    double x4 = line2.second.first;
    double y4 = line2.second.second;

    double denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1);
    if(denom==0.0)
        return false;
    
    double ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom;
    if(ua < 0 || ua > 1)
        return false;
    
    double ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom;
    if(ub < 0 || ub > 1)
        return false;

    return true;
}

double line_point_distance(pair<double,double> p1, pair<double,double> p2, pair<double,double> c)
{
    double x1 = p1.first;
    double y1 = p1.second;
    double x2 = p2.first;
    double y2 = p2.second;
    double x3 = c.first;
    double y3 = c.second;

    double px = x2 - x1;
    double py = y2 - y1;

    double something = px*px + py*py;
    double u = ((x3 - x1) * px + (y3 - y1) * py) / something;
    if(u > 1)
        u = 1;
    else if(u < 0)
        u = 0;
    double x = x1 + u * px;
    double y = y1 + u * py;
    double dx = x - x3;
    double dy = y - y3;
    return pow(dx*dx + dy*dy,0.5);
}


bool rectangle_circle_center(pair<double,double> lb, pair<double,double> rt, pair<double,double> c)
{
    double x1 = lb.first;
    double y1 = lb.second;
    double x2 = rt.first;
    double y2 = rt.second;
    double x3 = c.first;
    double y3 = c.second;

    double l_x = min(x1,x2);
    double r_x = max(x1,x2);
    double b_y = min(y1,y2);
    double t_y = max(y1,y2);
    if(l_x<=x3 && x3<=r_x && b_y<=y3 && y3<=t_y)
        return true;
    else
        return false;
}


bool rectangle_intersect_circle(pair<double,double> c, double radius, vector<pair<double,double>> rect)
{
    for(int i=0; i<4; i++){
        for(int j=1; j<5; j++){
            int tmp_i = i;
            int tmp_j = j%4;
            if(line_point_distance(rect[i], rect[j], c)<radius){
                return true;
            }
        }
    }
    return false;
}


Obstacle::Obstacle(pair<double,double> init_position, string obstacle_type, double radius, vector<pair<double,double>> polygon_points, bool is_static)
{
    obstacle_type = obstacle_type;
    if(obstacle_type=="circle"){
        x = init_position.first;
        y = init_position.second;
        radius = radius;
    }
    else if(obstacle_type=="rectangle"){
        polygon = polygon_points;
    }
    is_static = is_static;
}

bool Obstacle::feasibility_check(Obstacle target_object)
{
    if(obstacle_type=="circle"){
        if(target_object.obstacle_type=="circle"){
            if(pow(pow(x-target_object.x,2)+pow(y-target_object.y,2),0.5)<radius+target_object.radius){
                return false;
            }
            else{
                return true;
            }
        }
        else if(target_object.obstacle_type=="rectangle"){
            return !(rectangle_circle_center(target_object.polygon[0],target_object.polygon[2],make_pair(x,y)) || rectangle_intersect_circle(make_pair(x,y),radius,target_object.polygon));
        }
    }
    else if(obstacle_type=="rectangle"){
        if(target_object.obstacle_type=="circle"){
            return !(rectangle_circle_center(polygon[0],polygon[2],make_pair(target_object.x,target_object.y)) || rectangle_intersect_circle(make_pair(target_object.x,target_object.y),radius,polygon));
        }
    }
}

bool Obstacle::feasibility_check_point(pair<double,double> point)
{
    if(obstacle_type=="circle"){
        if(pow((x-point.first)*(x-point.first)+(y-point.second)*(y-point.second),0.5)<radius)
            return false;
        else
            return true;
    }
    else if(obstacle_type=="rectangle"){
        double x1 = polygon[0].first;
        double y1 = polygon[0].second;
        double x2 = polygon[3].first;
        double y2 = polygon[3].second;
        double x3 = point.first;
        double y3 = point.second;

        double l_x = min(x1,x2);
        double r_x = max(x1,x2);
        double b_y = min(y1,y2);
        double t_y = max(y1,y2);

        if(l_x<=x3 && x3<=r_x && b_y<=y3 && y3<=t_y)
            return false;
        else
            return true;
    }
}

Obstacles::Obstacles(int num, double obs_radius, vector<Obstacle> static_obstacles, double mwidth, double mheight, string movable_obstacle_type, string static_obstacle_type)
{
    radius = obs_radius;
    static_obstacles = static_obstacles;
    movable_obstacles = vector<Obstacle>();
    num = static_obstacles.size() + movable_obstacles.size();
    movable_num = movable_obstacles.size();
    static_num = static_obstacles.size();
    mwidth = mwidth;
    mheight = mheight;
    movable_obstacle_type = movable_obstacle_type;
    static_obstacle_type = static_obstacle_type;

    for(int idx=0; idx<movable_num; idx++){
        while(true){
            bool loop_sig = false;
            
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<double> dis(-0.5,0.5);
            double tmp_center_x = dis(gen)*mwidth;
            double tmp_center_y = dis(gen)*mheight;

            Obstacle tmp_ob = Obstacle(make_pair(tmp_center_x,tmp_center_y), "circle",
              radius, vector<pair<double,double>>(), false);
            
            for(int i=0; i<static_num; i++)
                if(!tmp_ob.feasibility_check(static_obstacles[i]))
                    loop_sig = true;

            if(loop_sig)
                continue;
            else{
                movable_obstacles.push_back(tmp_ob);
                break;
            }
        }
    }
}

bool Obstacles::feasibility_check(Obstacle object)
{
    for(int i=0; i<movable_num; i++)
        if(!movable_obstacles[i].feasibility_check(object))
            return false;

    for(int i=0; i<static_num; i++)
        if(!static_obstacles[i].feasibility_check(object))
            return false;
    return true;
}

bool Obstacles::feasibility_check_point(pair<double,double> point)
{
    for(int i=0; i<movable_num; i++)
        if(!movable_obstacles[i].feasibility_check_point(point))
            return false;

    for(int i=0; i<static_num; i++)
        if(!static_obstacles[i].feasibility_check_point(point))
            return false;
    return true;
}
