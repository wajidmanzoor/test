#pragma once

#include "common.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>


class Graph {
public:
  ui n;
  ui m;
  ui kmax;

  std::vector<ui> offset;
  std::vector<ui> neighbors;
  std::vector<ui> degree;
  std::vector<ui> core;
  std::vector<ui> corePeelSequence;
  std::string filePath;

public:
  Graph();
  Graph(std::string path);
  void getListingOrder(std::vector<ui> &arr);
  void coreDecompose(std::vector<ui> &arr);
};
