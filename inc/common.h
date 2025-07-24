#pragma once

#define miv(a, b) ((a) > (b) ? (b) : (a))
#define mav(a, b) ((a) < (b) ? (b) : (a))

#include <assert.h>
#include <string.h>

#include <cstdlib>
#include <fstream>
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits.h>
#include <map>
#include <mutex>
#include <set>
#include <sys/stat.h>
#include <utility>

#define NUM_BLKS_PER_SM 1
#define BLK_NUMS 216
#define BLK_DIM 1024
#define TOTAL_THREAD (BLK_NUMS * BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM / 32)
#define TOTAL_WARPS (BLK_NUMS * WARPS_EACH_BLK)

using namespace std;

typedef unsigned int ui;
typedef unsigned short ushort;
typedef unsigned char uchar;

const int INF = 1000000000;
const double DINF = 1000000000.0;

typedef struct {

  ui *offset;
  ui *neighbors;

  ui *degree;
  ui *cliqueDegree;

  int *cliqueCore;
  ui *cliqueCorePeelSequence;

  double *density;
  ui *motifCount;
} deviceGraphPointers;

typedef struct {

  ui *componentOffset;
  ui *components;
  ui *mapping;
  ui *reverseMapping;
} deviceComponentPointers;

typedef struct {
  ui *offset;
  ui *neighbors;
  ui *degree;
} deviceDAGpointer;

typedef struct {

  ui *trie;
  int *status;

} deviceCliquesPointer;

typedef struct {
  ui *partialCliquesPartition; // Virtual Partition  array to store partial
                               // cliques. each earp writes in its partition
  ui *partialCliques; // cliques from above are written in continues memeory
                      // here.
  ui *candidatesPartition; // Virtual partition array that stores candidates of
                           // each partial clique from partialCliquesPartition
  ui *candidates; // Stores Candidates of each partial clique in partialCliques.
  ui *offsetPartition; // Stores offset of candidates of each PC in
                       // partialCliquesPartition.
  ui *offset; // Stores offset of candidates of each PC in partialCliques.
  ui *validNeighMaskPartition; // Stores mask of valid neighbors of candidates
                               // of each clique in partialCliquesPartition.
                               // Bitwise mask that can be used to get its
                               // neighbors from neighbor array of DAG
  ui *validNeighMask;          // Same as above but for partialCliques.
  ui *count;                   // Number of partial cliques in each partiton.
  ui *temp; // Use for offset calculation. Stores total candidates in each Warp.
  ui *max;  // max degree in DAG.
} cliqueLevelDataPointer;

typedef struct {
  ui *mapping;
  ui *reverseMap;
  ui *offset;
  ui *neighbors;
  ui *cliqueDegree;
  double *density;
  ui *n;
  ui *m;
  ui *totalCliques;

} densestCorePointer;

typedef struct {
  ui *newOffset;
  ui *newNeighbors;
  ui *pruneStatus;

} devicePrunedNeighbors;

typedef struct {
  ui *height;
  double *excess;
  ui *offset;
  ui *neighbors;
  double *capacity;
  double *flow;
  ui *flowIndex;

} deviceFlowNetworkPointers;

extern deviceGraphPointers deviceGraph;
extern deviceDAGpointer deviceDAG;
extern cliqueLevelDataPointer levelData;
extern deviceCliquesPointer cliqueData;
extern densestCorePointer densestCore;
extern deviceComponentPointers conComp;
extern devicePrunedNeighbors prunedNeighbors;
extern deviceCliquesPointer finalCliqueData;
extern deviceFlowNetworkPointers flowNetwork;
