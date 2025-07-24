
#include "../inc/gpuMemoryAllocation.cuh"
#include "../utils/cuda_utils.cuh"

void memoryAllocationGraph(deviceGraphPointers &G, Graph &graph) {
  ui n = graph.n;
  ui m = graph.m;
  chkerr(cudaMalloc((void **)&(G.offset), (n + 1) * sizeof(ui)));
  chkerr(cudaMemcpy(G.offset, graph.offset.data(), (n + 1) * sizeof(ui),
                    cudaMemcpyHostToDevice));

  chkerr(cudaMalloc((void **)&(G.neighbors), (2 * m) * sizeof(ui)));
  chkerr(cudaMemcpy(G.neighbors, graph.neighbors.data(), (2 * m) * sizeof(ui),
                    cudaMemcpyHostToDevice));

  chkerr(cudaMalloc((void **)&(G.degree), n * sizeof(ui)));
  chkerr(cudaMemcpy(G.degree, graph.degree.data(), n * sizeof(ui),
                    cudaMemcpyHostToDevice));

  chkerr(cudaMalloc((void **)&(G.cliqueDegree), n * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(G.cliqueCore), n * sizeof(int)));
  chkerr(cudaMalloc((void **)&(G.cliqueCorePeelSequence), n * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(G.density), n * sizeof(double)));
  chkerr(cudaMalloc((void **)&(G.motifCount), n * sizeof(ui)));

  cudaDeviceSynchronize();
}

void memoryAllocationDAG(deviceDAGpointer &D, ui n, ui m) {
  chkerr(cudaMalloc((void **)&(D.offset), (n + 1) * sizeof(ui)));
  chkerr(cudaMemset(D.offset, 0, (n + 1) * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(D.neighbors), m * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(D.degree), n * sizeof(ui)));
  cudaDeviceSynchronize();
}

void memoryAllocationComponent(deviceComponentPointers &C, ui n, ui m) {
  chkerr(cudaMalloc((void **)&(C.componentOffset), (n + 1) * sizeof(ui)));
  chkerr(cudaMemset(C.componentOffset, 0, (n + 1) * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(C.components), n * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(C.mapping), n * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(C.reverseMapping), n * sizeof(ui)));
  cudaDeviceSynchronize();
}

void memoryAllocationTrie(deviceCliquesPointer &C, ui t, ui k) {
  chkerr(cudaMalloc((void **)&(C.trie), (t * k) * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(C.status), t * sizeof(int)));
  cudaDeviceSynchronize();
}

ui memoryAllocationlevelData(cliqueLevelDataPointer &L, ui k, ui pSize,
                             ui cpSize, ui maxDegree, ui totalWarps) {
  size_t partialSize = (size_t)totalWarps * pSize;
  size_t candidateSize = (size_t)totalWarps * cpSize;
  size_t offsetSize = (size_t)((pSize / (k - 1)) + 1) * totalWarps;
  ui maxBitMask = (maxDegree + 31) / 32;
  size_t maskSize = (size_t)cpSize * maxBitMask * totalWarps;
  ui max_ = partialSize / (k - 1);

  chkerr(cudaMalloc((void **)&(L.partialCliquesPartition),
                    partialSize * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(L.partialCliques), partialSize * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.candidatesPartition),
                    candidateSize * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(L.candidates), candidateSize * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.validNeighMaskPartition),
                    maskSize * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(L.validNeighMask),
                    maskSize * sizeof(ui)));

  chkerr(cudaMemset(L.validNeighMask, 0, maskSize * sizeof(ui)));
  chkerr(cudaMemset(L.validNeighMaskPartition, 0, maskSize * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.offsetPartition), offsetSize * sizeof(ui)));
  chkerr(cudaMemset(L.offsetPartition, 0, offsetSize * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.offset), offsetSize * sizeof(ui)));
  chkerr(cudaMemset(L.offset, 0, offsetSize * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.count), (totalWarps + 1) * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(L.temp), (totalWarps + 1) * sizeof(ui)));
  chkerr(cudaMemset(L.temp, 0, (totalWarps + 1) * sizeof(ui)));
  chkerr(cudaMemset(L.count, 0, (totalWarps + 1) * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(L.max), sizeof(ui)));
  chkerr(cudaMemcpy(L.max, &max_, sizeof(ui), cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();
  return maxBitMask;
}


void memoryAllocationDensestCore(densestCorePointer &C, ui n, ui density,
                                 ui totalCliques, ui graphsize) {

  chkerr(cudaMalloc((void **)&(C.mapping), n * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(C.offset), (n + 1) * sizeof(ui)));
  chkerr(cudaMemset(C.offset, 0, (n + 1) * sizeof(ui)));

  // neighbors will be allocated once we now the size

  chkerr(cudaMalloc((void **)&(C.cliqueDegree), n * sizeof(ui)));
  // chkerr(cudaMalloc((void**)&(C.cliqueCore), n * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(C.density), n * sizeof(double)));
  chkerr(
      cudaMemcpy(C.density, &density, sizeof(double), cudaMemcpyHostToDevice));

  chkerr(cudaMalloc((void **)&(C.n), sizeof(ui)));
  chkerr(cudaMemcpy(C.n, &n, sizeof(ui), cudaMemcpyHostToDevice));
  chkerr(cudaMalloc((void **)&(C.m), sizeof(ui)));
  chkerr(cudaMalloc((void **)&(C.totalCliques), sizeof(ui)));
  chkerr(cudaMemcpy(C.totalCliques, &totalCliques, sizeof(ui),
                    cudaMemcpyHostToDevice));

  chkerr(cudaMalloc((void **)&(C.reverseMap), graphsize * sizeof(ui)));
  cudaDeviceSynchronize();
}

void memoryAllocationPrunnedNeighbors(devicePrunedNeighbors &prunedNeighbors,
                                      ui n, ui m) {
  chkerr(
      cudaMalloc((void **)&(prunedNeighbors.newOffset), (n + 1) * sizeof(ui)));
  chkerr(cudaMemset(prunedNeighbors.newOffset, 0, (n + 1) * sizeof(ui)));

  chkerr(cudaMalloc((void **)&(prunedNeighbors.pruneStatus),
                    (2 * m) * sizeof(ui)));
  cudaDeviceSynchronize();
}

void memoryAllocationFlowNetwork(deviceFlowNetworkPointers &flowNetwork,
                                 ui vertexSize, ui neighborSize) {

  chkerr(cudaMalloc((void **)&(flowNetwork.height), (vertexSize) * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(flowNetwork.excess),
                    (vertexSize) * sizeof(double)));

  chkerr(cudaMalloc((void **)&(flowNetwork.offset),
                    (vertexSize + 1) * sizeof(ui)));
  chkerr(
      cudaMalloc((void **)&(flowNetwork.neighbors), neighborSize * sizeof(ui)));
  chkerr(cudaMalloc((void **)&(flowNetwork.capacity),
                    neighborSize * sizeof(double)));
  chkerr(
      cudaMalloc((void **)&(flowNetwork.flow), neighborSize * sizeof(double)));

  chkerr(
      cudaMalloc((void **)&(flowNetwork.flowIndex), neighborSize * sizeof(ui)));

  cudaDeviceSynchronize();
}

// Memory deallocation functions
void freeGraph(deviceGraphPointers &G) {
  chkerr(cudaFree(G.offset));
  chkerr(cudaFree(G.neighbors));
  chkerr(cudaFree(G.degree));
  chkerr(cudaFree(G.cliqueDegree));
  chkerr(cudaFree(G.cliqueCore));
  chkerr(cudaFree(G.cliqueCorePeelSequence));
  chkerr(cudaFree(G.density));
  chkerr(cudaFree(G.motifCount));
}

void freeComponents(deviceComponentPointers &C) {
  chkerr(cudaFree(C.componentOffset));
  chkerr(cudaFree(C.components));
  chkerr(cudaFree(C.mapping));
}

void freeTrie(deviceCliquesPointer &C) {
  chkerr(cudaFree(C.trie));
  chkerr(cudaFree(C.status));
}

void freeDAG(deviceDAGpointer &D) {
  chkerr(cudaFree(D.offset));
  chkerr(cudaFree(D.neighbors));
  chkerr(cudaFree(D.degree));
}

void freeLevelData(cliqueLevelDataPointer &L) {
  chkerr(cudaFree(L.partialCliques));
  chkerr(cudaFree(L.candidates));
  chkerr(cudaFree(L.offset));
  chkerr(cudaFree(L.validNeighMask));
  chkerr(cudaFree(L.count));
  chkerr(cudaFree(L.max));
  chkerr(cudaFree(L.partialCliquesPartition));
  chkerr(cudaFree(L.candidatesPartition));
  chkerr(cudaFree(L.offsetPartition));
  chkerr(cudaFree(L.validNeighMaskPartition));
  chkerr(cudaFree(L.temp));
}

void freeDensestCore(densestCorePointer &C) {
  chkerr(cudaFree(C.mapping));
  chkerr(cudaFree(C.offset));
  chkerr(cudaFree(C.neighbors));
  chkerr(cudaFree(C.density));
  chkerr(cudaFree(C.n));
  chkerr(cudaFree(C.m));
  chkerr(cudaFree(C.totalCliques));
  chkerr(cudaFree(C.cliqueDegree));
  // chkerr(cudaFree(C.cliqueCore));
}

void freePruneneighbors(devicePrunedNeighbors &P) {
  chkerr(cudaFree(P.newOffset));
  chkerr(cudaFree(P.newNeighbors));
  chkerr(cudaFree(P.pruneStatus));
}

void freeFlownetwork(deviceFlowNetworkPointers &F) {
  chkerr(cudaFree(F.height));
  chkerr(cudaFree(F.excess));
  chkerr(cudaFree(F.offset));
  chkerr(cudaFree(F.neighbors));
  chkerr(cudaFree(F.capacity));
  chkerr(cudaFree(F.flow));
  chkerr(cudaFree(F.flowIndex));
}
