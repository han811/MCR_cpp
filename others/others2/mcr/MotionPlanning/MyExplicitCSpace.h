#ifndef MY_EXPLICIT_CSPACE_H
#define MY_EXPLICIT_CSPACE_H

#include "MotionPlanning/ExplicitCSpace.h"
#include "misc/Circle2D.h"
#include "misc/Triangle2D.h"
#include "misc/AABB2D.h"
#include <vector>
using namespace MathGeometric;
using namespace std;

class MyExplicitCSpace : public ExplicitCSpace
{
public:
  MyExplicitCSpace();
  void Add(const Triangle2D& tri);
  void Add(const AABB2D& bbox);
  //void Add(const Polygon2D& poly);
  void Add(const Circle2D& sphere);

  //distance queries
  double ObstacleDistance(const Vector2& x) const;
  bool Overlap(const Circle2D& circle) const;

  virtual void Sample(Config& x);
  virtual void SampleNeighborhood(const Config& c,double r,Config& x);
  virtual bool IsFeasible(const Config& x);
  virtual EdgePlanner* LocalPlanner(const Config& a,const Config& b);
  EdgePlanner* LocalPlanner(const Config& a,const Config& b,int obstacle);

  virtual double Distance(const Config& x, const Config& y);
  virtual double ObstacleDistance(const Config& x) { return ObstacleDistance(Vector2(x[0],x[1])); }

  bool IsFeasible(const Config& a,int obstacle);
  int NumObstacles();

  bool euclideanSpace;
  double visibilityEpsilon;
  AABB2D domain;
  vector<AABB2D> aabbs;
  vector<Triangle2D> triangles;
  vector<Circle2D> circles;
};

#endif
