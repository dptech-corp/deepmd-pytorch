#include <torch/script.h>
#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
struct InputNlist
{
  int inum;
  int * ilist;
  int * numneigh;
  int ** firstneigh;
  InputNlist () 
      : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL)
      {};
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {};
  ~InputNlist(){};
};

struct NeighborListData {
  /// Array stores the core region atom's index
  std::vector<int> ilist;
  /// Array stores the core region atom's neighbor index
  //std::vector<std::vector<int>> jlist;
  int *jlist;
  /// Array stores the number of neighbors of core region atoms
  std::vector<int> numneigh;
  /// Array stores the the location of the first neighbor of core region atoms
  std::vector<int*> firstneigh;

 public:
  void copy_from_nlist(const InputNlist& inlist, int& max_num_neighbors,int nnei);
  //void make_inlist(InputNlist& inlist);
};
#endif