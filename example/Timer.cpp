#include "Timer.h"
#include <stdlib.h>

#ifdef WIN32
#define GETCURRENTTIME(x) x=timeGetTime()
#else
#define GETCURRENTTIME(x) gettimeofday(&x,NULL)
#endif //WIN32

// Sadly, timersub isn't defined in Solaris. :(
// So we use this instead. (added by Ryan)

#if defined (__SVR4) && defined (__sun)
#include "timersub.h"
#endif

Timer::Timer()
{
  Reset();
}

void Timer::Reset()
{
  GETCURRENTTIME(start);
  current=start;
}

long long Timer::ElapsedTicks()
{
  GETCURRENTTIME(current);
  return LastElapsedTicks();
}

long long Timer::LastElapsedTicks() const
{
#ifdef WIN32
  return current-start;
#else
  timeval delta;
  timersub(&current,&start,&delta);
  long long ticks = delta.tv_sec*1000 + delta.tv_usec/1000;
  return ticks;
#endif //WIN32
}
    
double Timer::ElapsedTime()
{
  GETCURRENTTIME(current);
  return LastElapsedTime();
}

double Timer::LastElapsedTime() const
{
#ifdef WIN32
  return double(current-start)/1000.0;
#else
  timeval delta;
  timersub(&current,&start,&delta);
  double secs=double(delta.tv_sec);
  secs += double(delta.tv_usec)/1000000.0;
  return secs;
#endif //WIN32
}
