#include "MyExplicitCSpace.h"
#include "misc/Random.h"
#include "misc/Sample.h"
#include "misc/Segment2D.h"
#include "misc/Line2D.h"
#include "MotionPlanning/EdgePlanner.h"
#include "misc/Miscellany.h"
#include <math.h>
using namespace ConstantHelper;

MyExplicitCSpace::MyExplicitCSpace()
{
  euclideanSpace=true;
  visibilityEpsilon = 0.01;
  domain.bmin.set(0,0);
  domain.bmax.set(1,1);
}

void MyExplicitCSpace::Add(const Triangle2D& tri)
{
  triangles.push_back(tri);
}
void MyExplicitCSpace::Add(const AABB2D& bbox)
{
  aabbs.push_back(bbox);
}
//void Add(const Polygon2D& poly);
void MyExplicitCSpace::Add(const Circle2D& sphere)
{
  circles.push_back(sphere);
}

double MyExplicitCSpace::ObstacleDistance(const Vector2& p) const
{
  double dmin = Inf;
  if(p.x < domain.bmin.x || p.x > domain.bmax.x) return 0;
  dmin = Min(dmin,p.x-domain.bmin.x);
  dmin = Min(dmin,domain.bmax.x-p.x);
  if(p.y < domain.bmin.y || p.y > domain.bmax.y) return 0;
  dmin = Min(dmin,p.y-domain.bmin.y);
  dmin = Min(dmin,domain.bmax.y-p.y);
  //now we're looking at d^2
  dmin *= dmin; 
  for(size_t i=0;i<triangles.size();i++)
    dmin = Min(dmin,triangles[i].closestPoint(p).distanceSquared(p));
  for(size_t i=0;i<aabbs.size();i++) {
    Vector2 q=p;
    if(q.x < aabbs[i].bmin.x) q.x = aabbs[i].bmin.x;
    if(q.x > aabbs[i].bmax.x) q.x = aabbs[i].bmax.x;
    if(q.y < aabbs[i].bmin.y) q.y = aabbs[i].bmin.y;
    if(q.y > aabbs[i].bmax.y) q.y = aabbs[i].bmax.y;
    dmin = Min(dmin,p.distanceSquared(q));
  }
  dmin = sqrt(dmin);
  //back to looking at d
  for(size_t i=0;i<circles.size();i++)
    dmin = Min(dmin,circles[i].distance(p));
  if(dmin < 0) return 0;
  return dmin;
}

bool MyExplicitCSpace::Overlap(const Circle2D& circle) const
{
  return ObstacleDistance(circle.center) <= circle.radius;
}

void MyExplicitCSpace::Sample(Config& x)
{
  bool sig = true;
  while(sig){
    sig = false;
    x.resize(2);
    x[0]=RandHelper::randWithRange(domain.bmin.x,domain.bmax.x);
    x[1]=RandHelper::randWithRange(domain.bmin.y,domain.bmax.y);
    for(int i=0; i<aabbs.size(); i++){
      Point2D temp_x;
      temp_x.x = x[0];
      temp_x.y = x[1];
      if(aabbs[i].contains(temp_x)){
        sig = true;
      }
    }
  }
}

void MyExplicitCSpace::SampleNeighborhood(const Config& c,double r,Config& x)
{
  x.resize(2);
  if(euclideanSpace) MathSample::SampleDisk(r,x[0],x[1]);
  else { x[0]=RandHelper::randWithRange(-r,r); x[1]=RandHelper::randWithRange(-r,r); }
  V::add(x, c, x);
}

bool MyExplicitCSpace::IsFeasible(const Config& x)
{
  Vector2 p(x[0],x[1]);
  if(!domain.contains(p)) return false;
  for(size_t i=0;i<triangles.size();i++)
    if(triangles[i].contains(p)) return false;
  for(size_t i=0;i<aabbs.size();i++)
    if(aabbs[i].contains(p)) return false;
  for(size_t i=0;i<circles.size();i++)
    if(circles[i].contains(p)) return false;
  return true;
}

class MyExplicitEdgePlanner : public EdgePlanner
{
public:
  MyExplicitEdgePlanner(const Config& _a,const Config& _b,MyExplicitCSpace* _space)
    :a(_a),b(_b),space(_space),done(false),failed(false)
  {
    s.a.set(_a[0],_a[1]);
    s.b.set(_b[0],_b[1]);
  }
  MyExplicitEdgePlanner(const Config& _a,const Config& _b,MyExplicitCSpace* _space,int obstacle)
    :a(_a),b(_b),space(_space),done(false),failed(false),obstacle(obstacle)
  {
    s.a.set(_a[0],_a[1]);
    s.b.set(_b[0],_b[1]);
  }
  virtual bool IsVisible() {
    int nAABB = space->aabbs.size();
    int nTRIANGLE = space->triangles.size();
    int nCIRCLE = space->circles.size();
    
    
    if(!space->domain.contains(s.a)) return false;
    if(!space->domain.contains(s.b)) return false;
    Plane2D p;
    p.setPoints(s.a,s.b);
    Line2D l;
    l.source = s.a;
    l.direction = s.b-s.a;
    

    if(nAABB-obstacle>0){
        if(s.intersects(space->aabbs[obstacle])) return false;
    }
    else if(nTRIANGLE+nAABB-obstacle>0){
        Segment2D s2;
        if(space->triangles[obstacle-nAABB].intersect(p,s2)) {
            double t1=l.closestPointParameter(s2.a);
            double t2=l.closestPointParameter(s2.b);
            if(Max(t1,t2) >= 0.0 && Min(t1,t2) <= 1.0)
                return false;
        }
    }
    else if(nCIRCLE+nTRIANGLE+nAABB-obstacle>0){
        double t1,t2;
        if(space->circles[obstacle-nAABB-nTRIANGLE].intersects(l,&t1,&t2)) {
            if(t2 >= 0.0 && t1 <= 1.0)
                return false;
        }
    }
    return true;

    // Segment2D s2;
    // for(size_t i=0;i<space->triangles.size();i++) {
    //   if(space->triangles[i].intersect(p,s2)) {
	// double t1=l.closestPointParameter(s2.a);
	// double t2=l.closestPointParameter(s2.b);
	// if(Max(t1,t2) >= 0.0 && Min(t1,t2) <= 1.0)
	//   return false;
    //   }
    // }
    // for(size_t i=0;i<space->aabbs.size();i++)
    //   if(s.intersects(space->aabbs[i])) return false;
    // double t1,t2;
    // for(size_t i=0;i<space->circles.size();i++) {
    //   if(space->circles[i].intersects(l,&t1,&t2)) {
	// if(t2 >= 0.0 && t1 <= 1.0)
	//   return false;
    //   }
    // }
    // return true;
  }
  virtual void Eval(double u,Config& x) const {
	Config tmp;
	V::multiply(a, One - u, x);
	V::multiply(b, u, tmp);
	V::add(tmp, x, x);
  }
  virtual const Config& Start() const { return a; }
  virtual const Config& Goal() const { return b; }
  virtual CSpace* Space() const { return space; }
  virtual EdgePlanner* Copy() const { return new MyExplicitEdgePlanner(a,b,space); }
  virtual EdgePlanner* ReverseCopy() const  { return new MyExplicitEdgePlanner(b,a,space); }

  //for incremental planners
  virtual double Priority() const { return s.a.distanceSquared(s.b); }
  virtual bool Plan() { failed = !IsVisible(); done = true; return !failed; }
  virtual bool Done() const {  return done; }
  virtual bool Failed() const {  return failed; }

  Config a,b;
  MyExplicitCSpace* space;
  Segment2D s;
  //used for incremental planner only
  bool done;
  bool failed;
  int obstacle;
};

EdgePlanner* MyExplicitCSpace::LocalPlanner(const Config& a,const Config& b)
{
  //return new BisectionEpsilonEdgePlanner(a,b,this,visibilityEpsilon);
  return new MyExplicitEdgePlanner(a,b,this);
}

EdgePlanner* MyExplicitCSpace::LocalPlanner(const Config& a,const Config& b,int obstacle)
{
  //return new BisectionEpsilonEdgePlanner(a,b,this,visibilityEpsilon);
  return new MyExplicitEdgePlanner(a,b,this,obstacle);
}



double MyExplicitCSpace::Distance(const Config& x, const Config& y)
{ 
	if(euclideanSpace) return V::euclideanDistance(x,y);
	else return V::lInfDistance(x,y);
}

bool MyExplicitCSpace::IsFeasible(const Config& a, int obstacle)
{
    int nAABB = aabbs.size();
    int nTRIANGLE = triangles.size();
    int nCIRCLE = circles.size();
  
    Vector2 p(a[0],a[1]);
  
    if(nAABB-obstacle>0){
        if(aabbs[obstacle].contains(p)) return false;
    }
    else if(nTRIANGLE+nAABB-obstacle>0){
        if(triangles[obstacle-nAABB].contains(p)) return false;
    }
    else if(nCIRCLE+nTRIANGLE+nAABB-obstacle){
        if(circles[obstacle-nAABB-nTRIANGLE].contains(p)) return false;
    }
    return true;
}

int MyExplicitCSpace::NumObstacles()
{
    return aabbs.size()+triangles.size()+circles.size();
}
