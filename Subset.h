#ifndef SUBSET_H
#define SUBSET_H

#include <vector>
#include <iostream>

struct Subset
{
  Subset(int maxItem=0);
  Subset(const Subset& s);
  Subset(const std::vector<bool>& bits);
  
  typedef std::vector<int>::iterator iterator;
  typedef std::vector<int>::const_iterator const_iterator;
  inline std::vector<int>::iterator begin() { return items.begin(); }
  inline std::vector<int>::iterator end() { return items.end(); }
  inline std::vector<int>::const_iterator begin() const { return items.begin(); }
  inline std::vector<int>::const_iterator end() const { return items.end(); }
  inline bool empty() const { return items.empty(); }
  inline size_t size() const {  return items.size(); }

  bool operator < (const Subset& s) const;
  bool operator > (const Subset& s) const;
  bool operator == (const Subset& s) const;
  bool operator != (const Subset& s) const;

  Subset operator + (const Subset& s) const;
  Subset operator - () const;
  Subset operator - (const Subset& s) const;
  Subset operator & (const Subset& s) const;

  void insert(int item);
  void insert_end(int item);
  void remove(int item);
  iterator find(int item);
  const_iterator find(int item) const;
  inline void erase(iterator it) { items.erase(it); }
  inline size_t count(int item) const { return (find(item)==end()?0:1); }
  bool is_subset(const Subset& s) const;

  int maxItem;
  std::vector<int> items;
};

std::ostream& operator << (std::ostream& out,const Subset& s);

#endif

