#include "MotionPlanning/MyExplicitCSpace.h"
#include "MotionPlanning/ExplainingPlanner.h"
#include "misc/Miscellany.h"


int main(void)
{
  MyExplicitCSpace myspace();

  /* Set up planner and set parameters (default values shown here) */
  ErrorExplainingPlanner planner(&myspace);
  planner.numConnections = 10;        //compute k-connected PRM
  planner.connectThreshold = ConstantHelper::Inf;     //haven't tested this setting much
  planner.expandDistance = 0.1;       //how far to expand the PRM toward a random configuration at each iteration
  planner.goalConnectThreshold = 0.5; //distance at which the planner attempts to connect configurations directly to the goal
  planner.usePathCover = true;        //keep this to true, otherwise performance can be quite bad
  planner.updatePathsComplete = false;//governs whether greedy or complete explanation set updates are used.  Can play with this.
  /* Set up planner */
  planner.Init(start,goal);

  /* Set up an explanation limit expansion schedule, up to 5000 iterations */
  vector<int> schedule(5);
  schedule[0] = 1000;
  schedule[1] = 2000;
  schedule[2] = 3000;
  schedule[3] = 4000;
  schedule[4] = 5000;
  
  /* Start planning */
  vector<int> path;
  Subset cover;
  planner.Plan(0,schedule,path,cover);

  //simple print (integers):
  cout<<"Best cover: "<<cover<<endl;

  //or pretty print (obstacle names):
  cout<<"Best cover:"<<endl;
  for(set<int>::const_iterator i=cover.items.begin();i!=cover.items.end();i++)
    cout<<"  "<<myspace.ObstacleName(*i)<<endl;
  return 0;
}