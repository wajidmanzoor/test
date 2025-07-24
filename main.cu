#include "./inc/common.h"
#include "./inc/graph.h"

#include "./inc/gpuMemoryAllocation.cuh"
#include "./inc/helpers.cuh"
#include "./utils/cuda_utils.cuh"
#include <thrust/count.h>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <iomanip>
#include <thrust/async/copy.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <chrono>

bool DEBUG = false;

void generateDAG(const Graph &graph, deviceGraphPointers &deviceGraph,
                 deviceDAGpointer &deviceDAG, vector<ui> listingOrder) {

  // Stores the Directed Acyclic Graph
  memoryAllocationDAG(deviceDAG, graph.n, graph.m);

  ui *listOrder;
  chkerr(cudaMalloc((void **)&(listOrder), graph.n * sizeof(ui)));
  chkerr(cudaMemcpy(listOrder, listingOrder.data(), graph.n * sizeof(ui),
                    cudaMemcpyHostToDevice));

  // Get out degree in DAG
  generateDegreeDAG<<<BLK_NUMS, BLK_DIM>>>(deviceGraph, deviceDAG, listOrder,
                                           graph.n, graph.m, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Degree of DAG");

  // copy out degree to offset
  chkerr(cudaMemset(deviceDAG.offset, 0, sizeof(ui)));
  chkerr(cudaMemcpy(deviceDAG.offset + 1, deviceDAG.degree,
                    (graph.n) * sizeof(ui), cudaMemcpyDeviceToDevice));

  // cummulative sum to get the offset of neighbors
  thrust::inclusive_scan(thrust::device_ptr<ui>(deviceDAG.offset),
                         thrust::device_ptr<ui>(deviceDAG.offset + graph.n + 1),
                         thrust::device_ptr<ui>(deviceDAG.offset));

  if (DEBUG) {
    ui *h_degree, *h_offset;
    h_degree = new ui[graph.n];
    h_offset = new ui[graph.n + 1];

    chkerr(cudaMemcpy(h_degree, deviceDAG.degree, graph.n * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(h_offset, deviceDAG.offset, (graph.n + 1) * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << endl << endl << "DAG DATA" << endl;
    cout << endl << "DAG" << endl << "Degree ";
    for (int i = 0; i < graph.n; i++) {
      cout << h_degree[i] << " ";
    }
    cout << endl << "offset ";
    for (int i = 0; i < graph.n + 1; i++) {
      cout << h_offset[i] << " ";
    }
    cout << endl;
  }

  // Writes neighbors of DAG based on the offset
  size_t sharedMemoryGenDagNeig = WARPS_EACH_BLK * sizeof(ui);
  generateNeighborDAG<<<BLK_NUMS, BLK_DIM, sharedMemoryGenDagNeig>>>(
      deviceGraph, deviceDAG, listOrder, graph.n, graph.m, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Neighbor of DAG");

  if (DEBUG) {
    ui *h_neighbors;
    h_neighbors = new ui[graph.m];
    chkerr(cudaMemcpy(h_neighbors, deviceDAG.neighbors, graph.m * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    cout << "neigh ";
    for (int i = 0; i < graph.m; i++) {
      cout << h_neighbors[i] << " ";
    }
    cout << endl;
  }

  chkerr(cudaFree(listOrder));
}

ui listAllCliques(const Graph &graph, deviceGraphPointers &deviceGraph,
                  deviceDAGpointer &deviceDAG, cliqueLevelDataPointer levelData,
                  ui k, ui pSize, ui cpSize) {
  /* Listing all k cliques of a graph using Bron–Kerbosch Algorithm.
   */

  // Get max out degree in DAG.
  thrust::device_ptr<ui> dev_degree(deviceDAG.degree);
  auto max_iter = thrust::max_element(dev_degree, dev_degree + graph.n);
  int maxDegree = *max_iter;

  // Allocates memory for intermediate results of the clique listing algorithm,
  // including partial cliques, candidate extensions for each partial clique,
  // and valid neighbors of each candidate.
  // Instead of storing actual valid neighbors, a bitmask is used to represent
  // valid neighbors. The maximum degree determines how many locations are
  // needed, with each location storing a 32-bit mask for up to 32 neighbors.

  ui maxBitMask = memoryAllocationlevelData(levelData, k, pSize, cpSize,
                                            maxDegree, TOTAL_WARPS);

  int level = 0;
  int iterK = k;

  // labels used to avoid duplicates.
  ui *labels;
  size_t numBits = static_cast<size_t>(graph.n) * TOTAL_WARPS;
  size_t numBytes = (numBits + 7) / 8;
  chkerr(cudaMalloc((void **)&(labels), (numBytes)));
  cudaMemset(labels, 0, numBytes);
  // thrust::device_ptr<ui> dev_labels(labels);
  // thrust::fill(dev_labels, dev_labels + total_size, iterK);

  chkerr(cudaMemcpy(deviceGraph.degree, graph.degree.data(),
                    graph.n * sizeof(ui), cudaMemcpyHostToDevice));
  chkerr(cudaMemset(levelData.partialCliquesPartition, 0,
                    (TOTAL_WARPS * pSize) * sizeof(ui)));
  cudaDeviceSynchronize();

  size_t sharedMemoryIntialClique = WARPS_EACH_BLK * sizeof(ui);

  // Generates initial partial cliques, their candidate extensions, and valid
  // neighbor masks. The data is stored in virtual partitions, with each warp
  // writing to a separate partition.

  cout << "psize " << pSize << endl;
  cout << "cpSize" << cpSize << endl;

  size_t partialSize1 = TOTAL_WARPS * pSize;
  size_t candidateSize = TOTAL_WARPS * cpSize;
  size_t offsetSize1 = ((pSize / (k - 1)) + 1) * TOTAL_WARPS;
  size_t maskSize = candidateSize * maxBitMask;

  listIntialCliques<<<BLK_NUMS, BLK_DIM, sharedMemoryIntialClique>>>(
      deviceDAG, levelData, labels, k, graph.n, pSize, cpSize, maxBitMask,
      level, TOTAL_WARPS, partialSize1, candidateSize, maskSize, offsetSize1);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Intial Partial Cliques");

  ui partialSize = TOTAL_WARPS * pSize;
  ui offsetSize = ((pSize / (k - 1)) + 1) * TOTAL_WARPS;
  ui offsetPartitionSize = ((pSize / (k - 1)) + 1);

  // Used to compute offsets so that partial cliques, candidates, and valid
  // neighbors can be copied from virtual partition to  contiguous array.
  createLevelDataOffset(levelData, offsetPartitionSize, TOTAL_WARPS);
  cudaDeviceSynchronize();

  // write partial cliques, candidates and valid neighbor bitmask from virtual
  // partition to single arrays.
  flushParitions<<<BLK_NUMS, BLK_DIM>>>(deviceDAG, levelData, pSize, cpSize, k,
                                        maxBitMask, level, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Flush Partition data structure");

  iterK--;
  level++;

  // Total number of partial cliques (tasks) for next level.
  int totalTasks;
  chkerr(cudaMemcpy(&totalTasks, levelData.count + TOTAL_WARPS, sizeof(ui),
                    cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  size_t sharedMemoryMid = WARPS_EACH_BLK * sizeof(ui);

  while (iterK > 2) {

    // thrust::device_ptr<ui> dev_labels(labels);
    // thrust::fill(dev_labels, dev_labels + graph.n * TOTAL_WARPS, iterK);
    cudaMemset(labels, 0, numBytes);

    chkerr(cudaMemset(levelData.count, 0, (TOTAL_WARPS + 1) * sizeof(ui)));
    chkerr(cudaMemset(levelData.temp, 0, (TOTAL_WARPS + 1) * sizeof(ui)));
    chkerr(cudaMemset(levelData.offsetPartition, 0, (offsetSize) * sizeof(ui)));
    chkerr(cudaMemset(levelData.validNeighMaskPartition, 0,
                      (partialSize * maxBitMask) * sizeof(ui)));
    cudaDeviceSynchronize();

    // Add verticies to partial cliques from initial kernel.
    listMidCliques<<<BLK_NUMS, BLK_DIM, sharedMemoryMid>>>(
        deviceDAG, levelData, labels, k, iterK, graph.n, graph.m, pSize, cpSize,
        maxBitMask, totalTasks, level, TOTAL_WARPS);
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR("Generate Mid Partial Cliques");

    // Used to compute offsets so that partial cliques, candidates, and valid
    // neighbors can be copied from virtual partition to  contiguous array.
    createLevelDataOffset(levelData, offsetPartitionSize, TOTAL_WARPS);

    chkerr(cudaMemset(levelData.offset, 0, offsetSize * sizeof(ui)));
    chkerr(cudaMemset(levelData.validNeighMask, 0,
                      partialSize * maxBitMask * sizeof(ui)));
    cudaDeviceSynchronize();

    // write partial cliques, candidates and valid neighbor bitmask from virtual
    // partition to single arrays.
    flushParitions<<<BLK_NUMS, BLK_DIM>>>(deviceDAG, levelData, pSize, cpSize,
                                          k, maxBitMask, level, TOTAL_WARPS);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Flush Partition data structure");

    // Get total partial cliques (tasks) for next level.
    chkerr(cudaMemcpy(&totalTasks, levelData.count + TOTAL_WARPS, sizeof(ui),
                      cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    iterK--;
    level++;
  }

  chkerr(cudaFree(labels));

  // Total Cliques
  ui *totalCliques;
  chkerr(cudaMalloc((void **)&totalCliques, sizeof(ui)));
  chkerr(cudaMemset(totalCliques, 0, sizeof(ui)));
  cudaDeviceSynchronize();

  countCliques<<<BLK_NUMS, BLK_DIM>>>(deviceDAG, levelData, totalCliques,
                                      maxBitMask, totalTasks, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Count Num Cliques");

  // Total Cliques
  ui tt;
  chkerr(cudaMemcpy(&tt, totalCliques, sizeof(ui), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  if (tt == 0) {
    freeLevelData(levelData);

    freeDAG(deviceDAG);
    return tt;
  }

  // Stores Final k-cliques of the graph.
  memoryAllocationTrie(cliqueData, tt, k);

  chkerr(cudaMemset(totalCliques, 0, sizeof(ui)));
  cudaDeviceSynchronize();

  size_t sharedMemoryFinal = WARPS_EACH_BLK * sizeof(ui);

  // Set status to of each clique to -2, representing not a valid clique.
  thrust::device_ptr<int> dev_ptr(cliqueData.status);
  thrust::fill(dev_ptr, dev_ptr + tt, -2);

  chkerr(cudaMemset(cliqueData.trie, 0, tt * k * sizeof(ui)));
  cudaDeviceSynchronize();

  if (iterK == 2) {

    // Write final k-cliques based on the partial cliques to global memory.
    writeFinalCliques<<<BLK_NUMS, BLK_DIM, sharedMemoryFinal>>>(
        deviceGraph, deviceDAG, levelData, cliqueData, totalCliques, k, iterK,
        graph.n, graph.m, pSize, cpSize, maxBitMask, tt, totalTasks, level,
        TOTAL_WARPS);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Generate Full Cliques");
  }

  if (DEBUG) {
    ui *h_cliques, *status;
    h_cliques = new ui[tt * k];
    status = new ui[tt];
    chkerr(cudaMemcpy(h_cliques, cliqueData.trie, k * tt * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(status, cliqueData.status, tt * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    cout << endl;

    cout << endl << "Cliques Data " << endl;
    for (int i = 0; i < k; i++) {
      cout << endl << "CL " << i << "  ";
      for (int j = 0; j < tt; j++) {
        cout << h_cliques[i * tt + j] << " ";
      }
    }
    cout << endl << "stat  ";
    for (int i = 0; i < tt; i++) {
      cout << status[i] << " ";
    }

    ui *h_cdegree;
    h_cdegree = new ui[graph.n];

    chkerr(cudaMemcpy(h_cdegree, deviceGraph.cliqueDegree, graph.n * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << endl << "Clique Degree" << endl;

    for (int i = 0; i < graph.n; i++) {
      cout << i << " ";
    }
    cout << endl;
    for (int i = 0; i < graph.n; i++) {
      cout << h_cdegree[i] << " ";
    }

    cout << endl;
  }

  // Actual total cliques in the graph.

  freeLevelData(levelData);

  freeDAG(deviceDAG);

  return tt;
}

void cliqueCoreDecompose(const Graph &graph, deviceGraphPointers &deviceGraph,
                         deviceCliquesPointer &cliqueData, ui &maxCore,
                         double &maxDensity, std::vector<ui> &coreSize,
                         ui &coreTotalCliques, ui glBufferSize, ui k,
                         ui totalCliques) {
  ui level = 0;
  ui count = 0;
  ui *globalCount = NULL;
  ui *bufTails = NULL;
  ui *glBuffers = NULL;

  // Total verticies that are removed.
  chkerr(cudaMalloc((void **)&(globalCount), sizeof(ui)));

  // stores the verticies that need to be removed in peeling algo.
  // Each warp stores in its virtual partition
  chkerr(
      cudaMalloc((void **)&(glBuffers), BLK_NUMS * glBufferSize * sizeof(ui)));

  // stores the end index of verticies in glBuffer for each warp
  chkerr(cudaMalloc((void **)&(bufTails), BLK_NUMS * sizeof(ui)));
  chkerr(cudaMemset(globalCount, 0, sizeof(ui)));
  cudaDeviceSynchronize();

  // set clique core to clique degree
  chkerr(cudaMemcpy(deviceGraph.cliqueCore, deviceGraph.cliqueDegree,
                    graph.n * sizeof(ui), cudaMemcpyDeviceToDevice));

  // total cliques yet to be removed
  thrust::device_ptr<int> dev_ptr(cliqueData.status);
  ui currentCliques = thrust::count(dev_ptr, dev_ptr + totalCliques, -1);

  double currentDensity =
      static_cast<double>(currentCliques) / (graph.n - count);

  maxDensity = currentDensity;
  maxCore = 0;
  coreTotalCliques = currentCliques;
  coreSize.push_back(graph.n);

  while (count < graph.n) {
    cudaMemset(bufTails, 0, sizeof(ui) * BLK_NUMS);

    // Select nodes whoes current degree is level, that means they should be
    // removed as part of the level core
    selectNodes<<<BLK_NUMS, BLK_DIM>>>(deviceGraph, bufTails, glBuffers,
                                       glBufferSize, graph.n, level);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Select Node Core Decompose");

    // Total number of verticies in buffer
    /*thrust::device_vector<ui> dev_vec1(bufTails, bufTails + BLK_NUMS);
    ui sum = thrust::reduce(dev_vec1.begin(), dev_vec1.end(), 0,
    thrust::plus<ui>()); cudaDeviceSynchronize();*/
    // TODO: ADD logic to switch between warp and block

    // Remove the verticies whose core value is current level and update clique
    // degrees
    processNodesByWarp<<<BLK_NUMS, BLK_DIM>>>(
        deviceGraph, cliqueData, bufTails, glBuffers, globalCount, glBufferSize,
        graph.n, level, k, totalCliques);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Process Node Core Decompose");

    // update the total verticies that are removed
    chkerr(cudaMemcpy(&count, globalCount, sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));

    level++;
    thrust::device_ptr<int> dev_ptr(cliqueData.status);

    // get the density of current core
    if ((graph.n - count) != 0) {
      currentCliques = thrust::count(dev_ptr, dev_ptr + totalCliques, -1);

      currentDensity = static_cast<double>(currentCliques) / (graph.n - count);

      // update max density
      if (currentDensity >= maxDensity) {
        maxDensity = currentDensity;
        maxCore = level;
        coreTotalCliques = currentCliques;
      }
      coreSize.push_back(graph.n - count);
    }

    cudaDeviceSynchronize();
  }
  cudaFree(globalCount);
  cudaFree(bufTails);
  cudaFree(glBuffers);
}

ui generateDensestCore(const Graph &graph, deviceGraphPointers &deviceGraph,
                       densestCorePointer &densestCore, ui coreSize,
                       ui coreTotalCliques, ui maxCore) {

  // stores the densest core in the graph
  memoryAllocationDensestCore(densestCore, coreSize, maxCore, coreTotalCliques,
                              graph.n);

  ui *globalCount;

  chkerr(cudaMalloc((void **)&globalCount, sizeof(ui)));
  chkerr(cudaMemset(globalCount, 0, sizeof(ui)));

  // Generates densest core, remaps its verticies and calculates the new offset.
  generateDensestCore<<<BLK_NUMS, BLK_DIM>>>(
      deviceGraph, densestCore, globalCount, graph.n, maxCore, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Densest Core");

  if (DEBUG) {
    ui *h_offset;
    h_offset = new ui[coreSize + 1];

    chkerr(cudaMemcpy(h_offset, densestCore.offset, (coreSize + 1) * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << endl << "offset b ";
    for (int i = 0; i <= coreSize; i++) {
      cout << h_offset[i] << " ";
    }
    cout << endl;
  }

  // cum sum to get the offset
  thrust::inclusive_scan(
      thrust::device_ptr<ui>(densestCore.offset),
      thrust::device_ptr<ui>(densestCore.offset + coreSize + 1),
      thrust::device_ptr<ui>(densestCore.offset));

  // debug
  if (DEBUG) {
    ui *h_mapping;
    h_mapping = new ui[coreSize];
    ui *h_offset;
    h_offset = new ui[coreSize + 1];
    chkerr(cudaMemcpy(h_mapping, densestCore.mapping, coreSize * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(h_offset, densestCore.offset, (coreSize + 1) * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << endl << "Densest core data " << endl;
    cout << endl << "mapping ";
    for (int i = 0; i < coreSize; i++) {
      cout << h_mapping[i] << " ";
    }
    cout << endl << "offset ";
    for (int i = 0; i <= coreSize; i++) {
      cout << h_offset[i] << " ";
    }
    cout << endl;
  }

  // Size of new neighbor list
  ui edgeCountCore;
  chkerr(cudaMemcpy(&edgeCountCore, densestCore.offset + coreSize, sizeof(ui),
                    cudaMemcpyDeviceToHost));

  chkerr(cudaMemcpy(densestCore.m, &edgeCountCore, sizeof(ui),
                    cudaMemcpyHostToDevice));
  chkerr(cudaMalloc((void **)&(densestCore.neighbors),
                    edgeCountCore * sizeof(ui)));

  // get the recerse mapping of verticies
  thrust::device_ptr<unsigned int> d_vertex_map_ptr(densestCore.mapping);
  thrust::device_ptr<unsigned int> d_reverse_map_ptr(densestCore.reverseMap);

  thrust::device_vector<unsigned int> d_indices(coreSize);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::scatter(d_indices.begin(), d_indices.end(), d_vertex_map_ptr,
                  d_reverse_map_ptr);

  size_t sharedMemoryGenNeighCore = WARPS_EACH_BLK * sizeof(ui);
  // Generate remaped neighbor list of the densest core
  generateNeighborDensestCore<<<BLK_NUMS, BLK_DIM, sharedMemoryGenNeighCore>>>(
      deviceGraph, densestCore, maxCore, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Densest Core neighbors");

  // Debug
  if (DEBUG) {
    ui *h_neighbors;
    h_neighbors = new ui[edgeCountCore];
    chkerr(cudaMemcpy(h_neighbors, densestCore.neighbors,
                      edgeCountCore * sizeof(ui), cudaMemcpyDeviceToHost));

    cout << endl << "neighbors ";
    for (int i = 0; i < edgeCountCore; i++) {
      cout << h_neighbors[i] << " ";
    }
    cout << endl;
  }

  chkerr(cudaFree(globalCount));

  return edgeCountCore;
}
ui prune(densestCorePointer &densestCore, deviceCliquesPointer &cliqueData,
         devicePrunedNeighbors &prunedNeighbors, ui vertexCount, ui edgecount,
         ui k, ui totalCliques, ui lowerBoundDensity) {

  // Allocate and initialize pruneStatus
  thrust::device_ptr<ui> d_pruneStatus(prunedNeighbors.pruneStatus);
  thrust::fill(d_pruneStatus, d_pruneStatus + edgecount, 1);

  // Kernel to determine pruning

  pruneEdges<<<BLK_NUMS, BLK_DIM>>>(densestCore, cliqueData,
                                    prunedNeighbors.pruneStatus, totalCliques,
                                    k, lowerBoundDensity);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Get prune status of each edge");

  if (DEBUG) {
    ui *h_pstatus = new ui[edgecount];
    chkerr(cudaMemcpy(h_pstatus, prunedNeighbors.pruneStatus,
                      edgecount * sizeof(ui), cudaMemcpyDeviceToHost));

    cout << endl << "Neigh Status" << endl;
    for (ui i = 0; i < edgecount; i++) {
      std::cout << h_pstatus[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_pstatus;
  }

  // Allocate and initialize newOffset

  // Kernel to generate out-degrees
  generateDegreeAfterPrune<<<BLK_NUMS, BLK_DIM>>>(
      densestCore, prunedNeighbors.pruneStatus, prunedNeighbors.newOffset,
      vertexCount, edgecount, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Degree after pruning");

  // Inclusive scan to build offset array
  thrust::inclusive_scan(
      thrust::device_pointer_cast(prunedNeighbors.newOffset),
      thrust::device_pointer_cast(prunedNeighbors.newOffset + vertexCount + 1),
      thrust::device_pointer_cast(prunedNeighbors.newOffset));

  // Get total number of remaining edges
  ui newEdgeCount;
  chkerr(cudaMemcpy(&newEdgeCount, prunedNeighbors.newOffset + vertexCount,
                    sizeof(ui), cudaMemcpyDeviceToHost));

  // Allocate memory for newNeighbors
  chkerr(cudaMalloc((void **)&(prunedNeighbors.newNeighbors),
                    newEdgeCount * sizeof(ui)));

  // Kernel to generate neighbors list
  size_t sharedMemoryGenNeig = WARPS_EACH_BLK * sizeof(ui);
  generateNeighborAfterPrune<<<BLK_NUMS, BLK_DIM, sharedMemoryGenNeig>>>(
      densestCore, prunedNeighbors.pruneStatus, prunedNeighbors.newOffset,
      prunedNeighbors.newNeighbors, vertexCount, edgecount, TOTAL_WARPS);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Generate Neighbor after prune");

  if (DEBUG) {
    ui *h_offset, *h_neigh;
    h_offset = new ui[vertexCount + 1];
    h_neigh = new ui[newEdgeCount];
    chkerr(cudaMemcpy(h_offset, prunedNeighbors.newOffset,
                      (vertexCount + 1) * sizeof(ui), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(h_neigh, prunedNeighbors.newNeighbors,
                      newEdgeCount * sizeof(ui), cudaMemcpyDeviceToHost));

    cout << endl << "Data After Pruning " << endl;
    cout << "in offset";
    for (ui i = 0; i < vertexCount + 1; i++) {
      cout << h_offset[i] << " ";
    }
    cout << endl;
    cout << "in neigh";
    for (ui i = 0; i < newEdgeCount; i++) {
      cout << h_neigh[i] << " ";
    }
    cout << endl;
  }

  return newEdgeCount;
}

int componentDecompose(deviceComponentPointers &conComp,
                       devicePrunedNeighbors &prunedNeighbors, ui vertexCount,
                       ui edgecount) {

  // flag indicating if any thread changed the component id of any vertex.
  ui *changed;
  chkerr(cudaMalloc((void **)&changed, sizeof(ui)));
  chkerr(cudaMemset(changed, 0, sizeof(ui)));

  // set component id of each vertex to its index.
  thrust::device_ptr<ui> components =
      thrust::device_pointer_cast(conComp.components);
  thrust::sequence(components, components + vertexCount);

  // int iter = 0;
  // can be used to put a limit on num iters
  ui hostChanged;
  do {
    chkerr(cudaMemset(changed, 0, sizeof(ui)));
    // changes component id of a vertex by convergence.
    componentDecomposek<<<BLK_NUMS, BLK_DIM>>>(
        conComp, prunedNeighbors, changed, vertexCount, edgecount, TOTAL_WARPS);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Coponenet Decompose");

    chkerr(
        cudaMemcpy(&hostChanged, changed, sizeof(ui), cudaMemcpyDeviceToHost));

  } while (hostChanged > 0);

  if (DEBUG) {
    ui *h_components;
    h_components = new ui[vertexCount];

    chkerr(cudaMemcpy(h_components, conComp.components,
                      vertexCount * sizeof(ui), cudaMemcpyDeviceToHost));
    cout << "comp 2 ";
    for (ui i = 0; i < vertexCount; i++) {
      cout << h_components[i] << " ";
    }
    cout << endl;
  }

  thrust::device_vector<ui> sorted_components(vertexCount);
  thrust::copy(components, components + vertexCount, sorted_components.begin());
  thrust::sort(sorted_components.begin(), sorted_components.end());

  // Total Unique components
  auto unique_end =
      thrust::unique(sorted_components.begin(), sorted_components.end());
  int totalComponents = unique_end - sorted_components.begin();
  cout << "Unique components: " << totalComponents << endl;

  // Create component offset array
  thrust::device_ptr<ui> componentOffsets(conComp.componentOffset);

  // Get component offset
  thrust::device_vector<ui> temp_sorted(vertexCount);
  thrust::copy(components, components + vertexCount, temp_sorted.begin());
  thrust::sort(temp_sorted.begin(), temp_sorted.end());

  thrust::lower_bound(temp_sorted.begin(), temp_sorted.end(),
                      sorted_components.begin(), unique_end, componentOffsets);
  componentOffsets[totalComponents] = vertexCount;

  // Renumber component id.
  thrust::lower_bound(sorted_components.begin(), unique_end, components,
                      components + vertexCount,
                      components // In-place remap
  );

  // Create component mapping
  thrust::sequence(thrust::device_pointer_cast(conComp.mapping),
                   thrust::device_pointer_cast(conComp.mapping + vertexCount));

  thrust::sort_by_key(components, components + vertexCount,
                      thrust::device_pointer_cast(conComp.mapping));

  if (DEBUG) {
    ui *h_components;
    h_components = new ui[vertexCount];
    ui *h_componentOffsets;
    h_componentOffsets = new ui[totalComponents + 1];
    chkerr(cudaMemcpy(h_componentOffsets, conComp.componentOffset,
                      (totalComponents + 1) * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    cout << endl << "Component Data " << endl;
    cout << "c off ";
    for (ui i = 0; i < totalComponents + 1; i++) {
      cout << h_componentOffsets[i] << " ";
    }
    cout << endl;
    chkerr(cudaMemcpy(h_components, conComp.components,
                      vertexCount * sizeof(ui), cudaMemcpyDeviceToHost));
    cout << "comp ";
    for (ui i = 0; i < vertexCount; i++) {
      cout << h_components[i] << " ";
    }
    cout << endl;

    ui *h_indicies;
    h_indicies = new ui[vertexCount];
    cout << " mapping " << endl;
    chkerr(cudaMemcpy(h_indicies, conComp.mapping, vertexCount * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    for (ui i = 0; i < vertexCount; i++) {
      cout << h_indicies[i] << " ";
    }
    cout << endl;
  }
  freePruneneighbors(prunedNeighbors);

  return totalComponents;
}
void dynamicExactAlgo(const Graph &graph, deviceGraphPointers &deviceGraph,
                      deviceFlowNetworkPointers &flowNetwork,
                      deviceComponentPointers &conComp,
                      deviceCliquesPointer &cliqueData,
                      deviceCliquesPointer &finalCliqueData, ui vertexCount,
                      ui totalComponents, ui totalCliques, ui k, ui maxCore,
                      ui partitionSize) {

  // Reverse mapping
  thrust::device_ptr<unsigned int> d_vertex_map_ptr(conComp.mapping);
  thrust::device_ptr<unsigned int> d_reverse_map_ptr(conComp.reverseMapping);
  thrust::device_vector<unsigned int> d_indices(vertexCount);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::scatter(d_indices.begin(), d_indices.end(), d_vertex_map_ptr,
                  d_reverse_map_ptr);

  if (DEBUG) {
    ui *rmap;
    rmap = new ui[vertexCount];
    chkerr(cudaMemcpy(rmap, conComp.reverseMapping, vertexCount * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << "R Map" << endl;
    for (ui i = 0; i < vertexCount; i++) {
      cout << rmap[i] << " ";
    }
    cout << endl;
  }

  // Component clique count
  ui *compCounter;
  chkerr(cudaMalloc((void **)&compCounter, (totalComponents + 1) * sizeof(ui)));
  chkerr(cudaMemset(compCounter, 0, (totalComponents + 1) * sizeof(ui)));

  getConnectedComponentStatus<<<BLK_NUMS, BLK_DIM>>>(
      conComp, cliqueData, densestCore, compCounter, totalCliques, k, maxCore,
      TOTAL_THREAD);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Calculate total cliques for each Component");

  thrust::inclusive_scan(
      thrust::device_pointer_cast(compCounter),
      thrust::device_pointer_cast(compCounter + totalComponents + 1),
      thrust::device_pointer_cast(compCounter));

  ui newTotaLCliques;
  chkerr(cudaMemcpy(&newTotaLCliques, compCounter + totalComponents, sizeof(ui),
                    cudaMemcpyDeviceToHost));

  memoryAllocationTrie(finalCliqueData, newTotaLCliques, k);

  ui *counter;
  chkerr(cudaMalloc((void **)&counter, totalComponents * sizeof(ui)));
  chkerr(cudaMemset(counter, 0, totalComponents * sizeof(ui)));

  rearrangeCliqueData<<<BLK_NUMS, BLK_DIM>>>(
      conComp, cliqueData, finalCliqueData, densestCore, compCounter, counter,
      totalCliques, k, newTotaLCliques, TOTAL_THREAD);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Rearrange Clique Data");
  freeTrie(cliqueData);

  if (DEBUG) {
    int *h_cliques = new int[newTotaLCliques * k];
    int *status = new int[newTotaLCliques];
    chkerr(cudaMemcpy(h_cliques, finalCliqueData.trie,
                      k * newTotaLCliques * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(status, finalCliqueData.status,
                      newTotaLCliques * sizeof(ui), cudaMemcpyDeviceToHost));
    for (int i = 0; i < k; i++) {
      std::cout << "\nCL " << i << "  ";
      for (int j = 0; j < newTotaLCliques; j++) {
        std::cout << h_cliques[i * newTotaLCliques + j] << " ";
      }
    }
    std::cout << "\nstat  ";
    for (int i = 0; i < newTotaLCliques; i++)
      std::cout << status[i] << " ";
    std::cout << std::endl;
    delete[] h_cliques;
    delete[] status;
  }

  double *bounds;
  chkerr(cudaMalloc((void **)&bounds, 2 * totalComponents * sizeof(double)));
  double *upperBound = bounds;
  double *lowerBound = bounds + totalComponents;

  int max_int = thrust::reduce(
      thrust::device_pointer_cast(deviceGraph.cliqueCore),
      thrust::device_pointer_cast(deviceGraph.cliqueCore + graph.n), 0,
      thrust::maximum<int>());

  double maxDensity1 = static_cast<double>(max_int);

  ui *flownetworkSize;
  ui *flownetworkNeighSize;
  chkerr(
      cudaMalloc((void **)&flownetworkSize, totalComponents * sizeof(double)));
  chkerr(cudaMalloc((void **)&flownetworkNeighSize,
                    totalComponents * sizeof(double)));

  getLbUbandSize<<<BLK_NUMS, BLK_DIM>>>(
      conComp, compCounter, lowerBound, upperBound, flownetworkSize,
      flownetworkNeighSize, totalComponents, k, maxDensity1);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Get UB LU and Size");

  if (DEBUG) {
    double *l_b, *u_b;
    l_b = new double[totalComponents];
    u_b = new double[totalComponents];
    chkerr(cudaMemcpy(l_b, lowerBound, totalComponents * sizeof(double),
                      cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(u_b, upperBound, totalComponents * sizeof(double),
                      cudaMemcpyDeviceToHost));

    cout << "lb ";
    for (ui i = 0; i < totalComponents; i++) {
      cout << l_b[i] << " ";
    }
    cout << endl;

    cout << "ub ";
    for (ui i = 0; i < totalComponents; i++) {
      cout << u_b[i] << " ";
    }
    cout << endl;
  }

  // Get iterator to max element
  auto max_iter = thrust::max_element(
      thrust::device_pointer_cast(lowerBound),
      thrust::device_pointer_cast(lowerBound + totalComponents));

  // Get max value by dereferencing iterator
  double max_lowerBound = *max_iter;

  // Get index by pointer arithmetic
  int argmax = max_iter - thrust::device_pointer_cast(lowerBound);

  cout << " max lb " << max_lowerBound << endl;

  thrust::fill(thrust::device_pointer_cast(lowerBound),
               thrust::device_pointer_cast(lowerBound + totalComponents),
               max_lowerBound);

  int iter = 0;
  ui *gpuConverged;
  chkerr(cudaMalloc((void **)&gpuConverged, sizeof(ui)));
  ui cpuConverged = 0;
  ui *changes;
  chkerr(cudaMalloc((void **)&changes, sizeof(ui)));

  ui *gpuSize;
  double *gpuMaxDensity;
  chkerr(cudaMalloc((void **)&gpuSize, sizeof(ui)));
  chkerr(cudaMalloc((void **)&gpuMaxDensity, sizeof(double)));

  ui from, to;
  chkerr(cudaMemcpy(&from, conComp.componentOffset + argmax, sizeof(ui),
                    cudaMemcpyDeviceToHost));
  chkerr(cudaMemcpy(&to, conComp.componentOffset + argmax + 1, sizeof(ui),
                    cudaMemcpyDeviceToHost));

  ui cpuSize = from - to;

  double cpuMaxDensity = max_lowerBound;

  ui DSD_size = 0;
  double DSD_density = 0;

  while (iter < totalComponents) {
    ui vertexSize, neighborSize;
    chkerr(cudaMemcpy(&vertexSize, flownetworkSize + iter, sizeof(ui),
                      cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(&neighborSize, flownetworkNeighSize + iter, sizeof(ui),
                      cudaMemcpyDeviceToHost));
    double ub, lb;
    chkerr(cudaMemcpy(&ub, upperBound + iter, sizeof(double),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(&lb, lowerBound + iter, sizeof(double),
                      cudaMemcpyDeviceToHost));
    if (ub > lb) {
      cout << "ub " << ub << " lb " << lb << endl;
      ui start, end, total;
      chkerr(cudaMemcpy(&start, conComp.componentOffset + iter, sizeof(ui),
                        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy(&end, conComp.componentOffset + iter + 1, sizeof(ui),
                        cudaMemcpyDeviceToHost));
      total = end - start;

      cout << "total size " << vertexSize << " vertex " << total << endl;

      memoryAllocationFlowNetwork(flowNetwork, vertexSize, neighborSize);

      thrust::fill(thrust::device_pointer_cast(flowNetwork.offset + 1),
                   thrust::device_pointer_cast(flowNetwork.offset + total + 1),
                   2);

      createFlowNetworkOffset<<<BLK_NUMS, BLK_DIM>>>(
          flowNetwork, conComp, finalCliqueData, compCounter, iter, k,
          newTotaLCliques);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR("Create Flow Network Offset");

      thrust::inclusive_scan(
          thrust::device_pointer_cast(flowNetwork.offset),
          thrust::device_pointer_cast(flowNetwork.offset + vertexSize + 1),
          thrust::device_pointer_cast(flowNetwork.offset));
      /*thrust::inclusive_scan(
          thrust::device_pointer_cast(flowNetwork.boffset),
          thrust::device_pointer_cast(flowNetwork.boffset + vertexSize + 1),
          thrust::device_pointer_cast(flowNetwork.boffset));*/
      ui *counter;
      chkerr(cudaMalloc((void **)&counter, total * sizeof(ui)));
      chkerr(cudaMemset(counter, 0, total * sizeof(ui)));

      createFlowNetwork<<<BLK_NUMS, BLK_DIM>>>(
          flowNetwork, conComp, finalCliqueData, compCounter, upperBound,
          lowerBound, counter, iter, k, newTotaLCliques);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR("Create Flow Network");
      chkerr(cudaFree(counter));
      if (DEBUG) {
        ui *offset, *neighbor, *index;
        offset = new ui[vertexSize + 1];
        neighbor = new ui[neighborSize];
        index = new ui[neighborSize];

        double *cap, *flow;
        cap = new double[neighborSize];
        flow = new double[neighborSize];

        cudaMemcpy(offset, flowNetwork.offset, (vertexSize + 1) * sizeof(ui),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(neighbor, flowNetwork.neighbors, (neighborSize) * sizeof(ui),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(index, flowNetwork.flowIndex, (neighborSize) * sizeof(ui),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cap, flowNetwork.capacity, (neighborSize) * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, flowNetwork.flow, (neighborSize) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cout << "offset ";
        for (ui i = 0; i < vertexSize + 1; i++) {
          cout << offset[i] << " ";
        }
        cout << endl;

        cout << "Neigh ";
        for (ui i = 0; i < neighborSize; i++) {
          cout << neighbor[i] << " ";
        }
        cout << endl;

        cout << "index ";
        for (ui i = 0; i < neighborSize; i++) {
          cout << index[i] << " ";
        }
        cout << endl;

        cout << "cap ";
        for (ui i = 0; i < neighborSize; i++) {
          cout << cap[i] << " ";
        }
        cout << endl;

        cout << "flow ";
        for (ui i = 0; i < neighborSize; i++) {
          cout << flow[i] << " ";
        }
        cout << endl;
      }
      double *totalExcess;
      chkerr(cudaMalloc((void **)&totalExcess, sizeof(double)));

      double hostSourceExcess, hostSinkExcess, hostTotalExcess;

      int device = -1;
      cudaGetDevice(&device);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      dim3 num_blocks(deviceProp.multiProcessorCount * NUM_BLKS_PER_SM);
      dim3 block_size(BLK_DIM);
      size_t sharedMemSize = 3 * block_size.x * sizeof(int);
      ui *activeNodes;
      chkerr(cudaMalloc((void **)&activeNodes, neighborSize * sizeof(ui)));

      void *pushRelabelKernelArgs[] = {&flowNetwork, &conComp,     &compCounter,
                                       &totalExcess, &activeNodes, &iter};

      void *globalRelabelKernelArgs[] = {&flowNetwork, &conComp, &compCounter,
                                         &changes,     &k,       &iter};

      void *updateFlownetworkKernelArgs[] = {
          &flowNetwork,   &conComp,    &finalCliqueData, &compCounter,
          &upperBound,    &lowerBound, &gpuConverged,    &gpuSize,
          &gpuMaxDensity, &k,          &newTotaLCliques, &iter};

      chkerr(cudaMemset(gpuConverged, 0, sizeof(ui)));
      cpuConverged = 0;
      while (!cpuConverged) {
        chkerr(cudaMemset(totalExcess, 0, sizeof(double)));
        preFlow<<<BLK_NUMS, BLK_DIM>>>(flowNetwork, conComp, compCounter,
                                       totalExcess, iter);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("Preflow send");
        if (DEBUG) {
          ui *offset, *neighbor, *index;
          offset = new ui[vertexSize + 1];
          neighbor = new ui[neighborSize];
          index = new ui[neighborSize];

          double *cap, *flow;
          cap = new double[neighborSize];
          flow = new double[neighborSize];

          ui *height;
          double *excess;
          height = new ui[vertexSize];
          excess = new double[vertexSize];

          cudaMemcpy(offset, flowNetwork.offset, (vertexSize + 1) * sizeof(ui),
                     cudaMemcpyDeviceToHost);

          cudaMemcpy(neighbor, flowNetwork.neighbors,
                     (neighborSize) * sizeof(ui), cudaMemcpyDeviceToHost);

          cudaMemcpy(index, flowNetwork.flowIndex, (neighborSize) * sizeof(ui),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy(cap, flowNetwork.capacity, (neighborSize) * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy(flow, flowNetwork.flow, (neighborSize) * sizeof(double),
                     cudaMemcpyDeviceToHost);

          cudaMemcpy(height, flowNetwork.height, (vertexSize) * sizeof(ui),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy(excess, flowNetwork.excess, (vertexSize) * sizeof(double),
                     cudaMemcpyDeviceToHost);

          cout << "offset ";
          for (ui i = 0; i < vertexSize + 1; i++) {
            cout << offset[i] << " ";
          }
          cout << endl;

          cout << "Neigh ";
          for (ui i = 0; i < neighborSize; i++) {
            cout << neighbor[i] << " ";
          }
          cout << endl;

          cout << "index ";
          for (ui i = 0; i < neighborSize; i++) {
            cout << index[i] << " ";
          }
          cout << endl;

          cout << "cap ";
          for (ui i = 0; i < neighborSize; i++) {
            cout << cap[i] << " ";
          }
          cout << endl;

          cout << "flow ";
          for (ui i = 0; i < neighborSize; i++) {
            cout << flow[i] << " ";
          }
          cout << endl;

          cout << "height ";
          for (ui i = 0; i < vertexSize; i++) {
            cout << height[i] << " ";
          }
          cout << endl;

          cout << "excess ";
          for (ui i = 0; i < vertexSize; i++) {
            cout << excess[i] << " ";
          }
          cout << endl;

          double te;
          cudaMemcpy(&te, totalExcess, sizeof(double), cudaMemcpyDeviceToHost);
          cout << "TOTAL EXCESS " << te << endl;
        }

        chkerr(cudaMemcpy(&hostSourceExcess,
                          flowNetwork.excess + vertexSize - 2, sizeof(double),
                          cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&hostSinkExcess, flowNetwork.excess + vertexSize - 1,
                          sizeof(double), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&hostTotalExcess, totalExcess, sizeof(double),
                          cudaMemcpyDeviceToHost));
        double epsilon = 1e-2; // Tolerance = 0.01
        while (abs((hostSourceExcess + hostSinkExcess) - hostTotalExcess) >
               epsilon) {

          cudaLaunchCooperativeKernel((void *)pushRelabel, num_blocks,
                                      block_size, pushRelabelKernelArgs,
                                      sharedMemSize, 0);
          cudaDeviceSynchronize();
          CUDA_CHECK_ERROR("Push Relabel");
          if (DEBUG) {
            cout << "after relabel" << endl;
            ui *offset, *neighbor, *index;
            offset = new ui[vertexSize + 1];
            neighbor = new ui[neighborSize];
            index = new ui[neighborSize];

            double *cap, *flow;
            cap = new double[neighborSize];
            flow = new double[neighborSize];

            ui *height;
            double *excess;
            height = new ui[vertexSize];
            excess = new double[vertexSize];

            cudaMemcpy(offset, flowNetwork.offset,
                       (vertexSize + 1) * sizeof(ui), cudaMemcpyDeviceToHost);

            cudaMemcpy(neighbor, flowNetwork.neighbors,
                       (neighborSize) * sizeof(ui), cudaMemcpyDeviceToHost);

            cudaMemcpy(index, flowNetwork.flowIndex,
                       (neighborSize) * sizeof(ui), cudaMemcpyDeviceToHost);
            cudaMemcpy(cap, flowNetwork.capacity,
                       (neighborSize) * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(flow, flowNetwork.flow, (neighborSize) * sizeof(double),
                       cudaMemcpyDeviceToHost);

            cudaMemcpy(height, flowNetwork.height, (vertexSize) * sizeof(ui),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(excess, flowNetwork.excess,
                       (vertexSize) * sizeof(double), cudaMemcpyDeviceToHost);

            cout << "offset ";
            for (ui i = 0; i < vertexSize + 1; i++) {
              cout << offset[i] << " ";
            }
            cout << endl;

            cout << "Neigh ";
            for (ui i = 0; i < neighborSize; i++) {
              cout << neighbor[i] << " ";
            }
            cout << endl;

            cout << "index ";
            for (ui i = 0; i < neighborSize; i++) {
              cout << index[i] << " ";
            }
            cout << endl;

            cout << "cap ";
            for (ui i = 0; i < neighborSize; i++) {
              cout << cap[i] << " ";
            }
            cout << endl;

            cout << "flow ";
            for (ui i = 0; i < neighborSize; i++) {
              cout << flow[i] << " ";
            }
            cout << endl;

            cout << "height ";
            for (ui i = 0; i < vertexSize; i++) {
              cout << height[i] << " ";
            }
            cout << endl;

            cout << "excess ";
            for (ui i = 0; i < vertexSize; i++) {
              cout << excess[i] << " ";
            }
            cout << endl;

            double te;
            cudaMemcpy(&te, totalExcess, sizeof(double),
                       cudaMemcpyDeviceToHost);
            cout << "TOTAL EXCESS " << te << endl;
          }
          chkerr(cudaMemset(changes, 0, sizeof(ui)));
          cudaLaunchCooperativeKernel((void *)globalRelabel, num_blocks,
                                      block_size, globalRelabelKernelArgs, 0,
                                      nullptr);
          cudaDeviceSynchronize();
          CUDA_CHECK_ERROR("Global Relabel");

          chkerr(cudaMemcpy(&hostSourceExcess,
                            flowNetwork.excess + vertexSize - 2, sizeof(double),
                            cudaMemcpyDeviceToHost));
          chkerr(cudaMemcpy(&hostSinkExcess,
                            flowNetwork.excess + vertexSize - 1, sizeof(double),
                            cudaMemcpyDeviceToHost));
          chkerr(cudaMemcpy(&hostTotalExcess, totalExcess, sizeof(double),
                            cudaMemcpyDeviceToHost));

          // break;
        }
        chkerr(cudaMemset(gpuMaxDensity, 0, sizeof(double)));
        chkerr(cudaMemset(gpuSize, 0, sizeof(ui)));

        cudaLaunchCooperativeKernel((void *)updateFlownetwork, num_blocks,
                                    block_size, updateFlownetworkKernelArgs, 0,
                                    nullptr);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("Update  flow Network");
        chkerr(cudaMemcpy(flowNetwork.flow, flowNetwork.capacity,
                          (neighborSize) * sizeof(double),
                          cudaMemcpyDeviceToDevice));

        chkerr(cudaMemcpy(&cpuConverged, gpuConverged, sizeof(ui),
                          cudaMemcpyDeviceToHost));
        chkerr(
            cudaMemcpy(&cpuSize, gpuSize, sizeof(ui), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&cpuMaxDensity, gpuMaxDensity, sizeof(double),
                          cudaMemcpyDeviceToHost));
        // cout<<" CPU MAX DENSITY"<<cpuMaxDensity<<endl;

        if (cpuMaxDensity >= DSD_density) {
          DSD_density = cpuMaxDensity;
          DSD_size = cpuSize;
        }
      }

      cudaFree(totalExcess);
    }
    freeFlownetwork(flowNetwork);
    iter++;
  }
  cout << "Max Density " << DSD_density << " Size " << DSD_size << endl;
}

int main(int argc, const char *argv[]) {
  if (argc != 7) {
    cout << "Server wrong input parameters!" << endl;
    exit(1);
  }

  cout << "TOTAL WARPS" << TOTAL_WARPS << endl;

  string filepath =
      argv[1]; //  Path to the graph file. The graph should be represented as
               //  an adjacency list with space separators
  // string motifPath = argv[2]; // Path to motif file. The motif should be
  // represented as edge list with space sperators. Not used yet
  ui k = atoi(argv[2]);      // The clique we are intrested in.
  ui pSize = atoi(argv[3]);  // Virtual Partition size for storing partial
                             // cliques in Listing Algorithm
  ui cpSize = atoi(argv[4]); // Virtual Partition size for storing Candidates
                             // of PC in Listing Algorithm
  ui glBufferSize =
      atoi(argv[5]); // Buffer to store the vertices that need to be removed
                     // in clique core decompose peeling algorithm
  ui partitionSize = atoi(argv[6]); // Virtual Partition to store the active
                                    // node of each flownetwork
  // ui t = atoi(argv[8]);             // Total Number of cliques.

  if (DEBUG) {
    cout << "filepath: " << filepath << endl;
    // cout << "motifPath: " << motifPath << endl;
    cout << "k: " << k << endl;
    cout << "pSize: " << pSize << endl;
    cout << "cpSize: " << cpSize << endl;
  }

  // Read Graph as a adcajency List.
  Graph graph = Graph(filepath);

  // Print Graph
  if (DEBUG) {
    cout << "Graph Data " << endl;
    cout << "Graph" << endl << "Offset: ";
    for (int i = 0; i < (graph.n + 1); i++) {
      cout << graph.offset[i] << " ";
    }
    cout << endl << "Neighbors: ";
    for (int i = 0; i < 2 * graph.m; i++) {
      cout << graph.neighbors[i] << " ";
    }
    cout << endl;
    cout << "Degree: ";
    for (int i = 0; i < graph.n; i++) {
      cout << graph.degree[i] << " ";
    }
    cout << endl;
  }

  // Stores the listing order based on core values:
  // vertices with higher core values are assigned lower (better) ranks.
  vector<ui> listingOrder;
  listingOrder.resize(graph.n);
  graph.getListingOrder(listingOrder);

  if (DEBUG) {
    cout << endl << endl << "Listing Order: ";
    for (int i = 0; i < graph.n; i++) {
      cout << listingOrder[i] << " ";
    }
    cout << endl << "Core ";
    for (int i = 0; i < graph.n; i++) {
      cout << graph.core[i] << " ";
    }
    cout << endl << "Peel Seq ";
    for (int i = 0; i < graph.n; i++) {
      cout << graph.corePeelSequence[i] << " ";
    }
    cout << endl;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Structure to store the graph on device
  memoryAllocationGraph(deviceGraph, graph);

  // Generates the DAG based on the listing order.
  // Only includes edges from a vertex with a lower listing order to one with
  // a higher listing order.
  generateDAG(graph, deviceGraph, deviceDAG, listingOrder);

  ui totalCliques;
  ui coreTotalCliques, maxCore;
  double maxDensity;

  std::vector<ui> coreSize;

  if (k == 2) {
    chkerr(cudaMemcpy(&totalCliques, deviceDAG.offset + graph.n, sizeof(ui),
                      cudaMemcpyDeviceToHost));
    // totalCliques = deviceDAG.offset[graph.n + 1];

    memoryAllocationTrie(cliqueData, totalCliques, k);

    cout << "Total cliques " << totalCliques << endl;

    chkerr(cudaMemcpy(deviceGraph.cliqueCore, graph.core.data(),
                      graph.n * sizeof(int), cudaMemcpyHostToDevice));

    ui *cliqueCount;
    chkerr(cudaMalloc((void **)&(cliqueCount), (graph.kmax + 1) * sizeof(ui)));
    chkerr(cudaMemset(cliqueCount, 0, (graph.kmax + 1) * sizeof(ui)));
    size_t sharedMemoryWriteEdgesCliques = WARPS_EACH_BLK * sizeof(ui);

    writeEdgeCliques<<<BLK_NUMS, BLK_DIM, sharedMemoryWriteEdgesCliques>>>(
        deviceGraph, deviceDAG, cliqueData, cliqueCount, graph.n, graph.m,
        pSize, totalCliques, TOTAL_WARPS);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Write Edge Clique");
    ui *cc;
    cc = new ui[graph.kmax + 1];
    chkerr(cudaMemcpy(cc, cliqueCount, (graph.kmax + 1) * sizeof(ui),
                      cudaMemcpyDeviceToHost));


    coreTotalCliques = totalCliques;
    maxCore = 0;
    maxDensity = static_cast<double>(coreTotalCliques) / graph.n;

    coreSize.push_back(graph.n);

    thrust::device_ptr<int> dev_ptr(deviceGraph.cliqueCore);

    // Lambda captures 'i' (must be marked __host__ __device__)
    cout << "max " << maxCore << " total Cliques " << coreTotalCliques << endl;

    double currentMax;

    for (ui i = 1; i <= graph.kmax; i++) {
      auto predicate = [i] __host__ __device__(int x) { return x >= i; };

      ui count = thrust::count_if(dev_ptr, dev_ptr + graph.n, predicate);
      // cout << "Count " << count << endl;
      coreSize.push_back(count);
      
      coreTotalCliques = coreTotalCliques - cc[i - 1];
      currentMax = static_cast<double>(coreTotalCliques) / count;
      // coreSize.push_back(count);
      if (currentMax >= maxDensity) {
        maxDensity = currentMax;
        maxCore = i;
      }
    }
    cout << "max " << maxCore << " total Cliques " << coreTotalCliques
         << " count " << coreSize[maxCore] << " Max density " << maxDensity
         << endl;

  } else {
    totalCliques = listAllCliques(graph, deviceGraph, deviceDAG, levelData, k,
                                  pSize, cpSize);
    cliqueCoreDecompose(graph, deviceGraph, cliqueData, maxCore, maxDensity,
                        coreSize, coreTotalCliques, glBufferSize, k,
                        totalCliques);
  }
  if (totalCliques == 0) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    freeGraph(deviceGraph);
    cout << "Max Density " << 0 << " Size " << 0 << endl;
    std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;
    return 0;
  }

  if (DEBUG) {
    cout << endl << "Clique data after core decompose " << endl;
    int *h_cliques, *status;
    h_cliques = new int[totalCliques * k];
    status = new int[totalCliques];
    chkerr(cudaMemcpy(h_cliques, cliqueData.trie, k * totalCliques * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(status, cliqueData.status, totalCliques * sizeof(ui),
                      cudaMemcpyDeviceToHost));
    for (int i = 0; i < k; i++) {
      cout << endl << "CL " << i << "  ";
      for (int j = 0; j < totalCliques; j++) {
        cout << h_cliques[i * totalCliques + j] << " ";
      }
    }
    cout << endl << "stat  ";
    for (int i = 0; i < totalCliques; i++) {
      cout << status[i] << " ";
    }
    cout << endl;

    int *cores;
    cores = new int[graph.n];
    chkerr(cudaMemcpy(cores, deviceGraph.cliqueCore, graph.n * sizeof(ui),
                      cudaMemcpyDeviceToHost));

    cout << "Core Values " << endl;
    for (int i = 0; i < graph.n; i++) {
      cout << cores[i] << " ";
    }
    cout << endl;
  }

  // ui lowerBoundDensity = maxCore;

  // Find the densest core in the graph.
  ui k_prime = std::ceil(maxDensity);

  cout << "k prime " << k_prime << endl;

  cout<<"core ar size "<<coreSize.size()<<endl;

  ui coresize = coreSize[k_prime];

  ui edgecount = generateDensestCore(graph, deviceGraph, densestCore, coresize,
                                     coreTotalCliques, k_prime);

  ui vertexCount;
  chkerr(cudaMemcpy(&vertexCount, densestCore.n, sizeof(ui),
                    cudaMemcpyDeviceToHost));

  // Structure to store the prunned neighbors
  memoryAllocationPrunnedNeighbors(prunedNeighbors, vertexCount, edgecount);

  // Prune invalid edges i.e. edges that are not part of any clique.
  ui newEdgeCount = prune(densestCore, cliqueData, prunedNeighbors, vertexCount,
                          edgecount, k, totalCliques, k_prime);

  freeDensestCore(densestCore);

  // Structure to store connected components
  memoryAllocationComponent(conComp, vertexCount, newEdgeCount);

  // Decomposes the graph into connected components and remaps vertex IDs
  // within each component.
  ui totalComponents =
      componentDecompose(conComp, prunedNeighbors, vertexCount, newEdgeCount);

  // Dynamic exact

  dynamicExactAlgo(graph, deviceGraph, flowNetwork, conComp, cliqueData,
                   finalCliqueData, vertexCount, totalComponents, totalCliques,
                   k, k_prime, partitionSize);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end - start;
  freeComponents(conComp);
  freeGraph(deviceGraph);
  freeTrie(finalCliqueData);
  std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;

  return 0;
}