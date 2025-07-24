#include "../inc/helpers.cuh"
#include "../utils/cuda_utils.cuh"

#define ASSERT_DEBUG(cond, fmt, ...)                     \
  if (!(cond)) {                                         \
    printf("ASSERT FAILED at %s:%d: " fmt,               \
           __FILE__, __LINE__, ##__VA_ARGS__);           \
    assert(false);                                       \
  }

using namespace cooperative_groups;
namespace cg = cooperative_groups;

__device__ double fact(ui k) {
  double res = 1;
  int i = k;
  while (i > 1) {
    res = res * i;
    i--;
  }
  return res;
}

inline __device__ void
scan_active_vertices(int totalFlow, ui source, ui sink,
                     deviceFlowNetworkPointers flowNetwork, ui *activeNodes) {
  unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_group grid = this_grid();

  /* Initialize the avq_size */
  if (idx == 0) {
    globalCounter = 0;
  }
  grid.sync();

  /* Stride scan the V set */
  for (int u = idx; u < totalFlow; u += blockDim.x * gridDim.x) {
    if (flowNetwork.excess[u] > 0 && flowNetwork.height[u] < totalFlow &&
        u != source && u != sink) {
      activeNodes[atomicAdd(&globalCounter, 1)] = u;
    }
  }
}

template <unsigned int tileSize>
inline __device__ int
tiled_search_neighbor(cg::thread_block_tile<tileSize> tile, int pos,
                      int *sheight, int *svid, int *svidx, int *v_index,
                      int totalFlow, ui source, ui sink,
                      deviceFlowNetworkPointers flowNetwork, ui *activeNodes) {
  unsigned int idx = tile.thread_rank(); // 0~31
  ui u = activeNodes[pos];
  int degree = flowNetwork.offset[u + 1] - flowNetwork.offset[u];
  int num_iters = (int)ceilf((float)degree / (float)tileSize);

  int minH = INF;
  int minV = -1;

  /* Initialize the shared memory */
  sheight[threadIdx.x] = INF;
  svid[threadIdx.x] = -1;
  svidx[threadIdx.x] = -2;
  tile.sync();

  for (int i = 0; i < num_iters; i++) {
    ui v_pos, v;
    if (i * tileSize + idx < degree) {
      v_pos = flowNetwork.offset[u] + i * tileSize + idx;
      v = flowNetwork.neighbors[v_pos];
      if ((flowNetwork.flow[v_pos] > 0) && (v != source)) {
        sheight[threadIdx.x] = flowNetwork.height[v];
        svid[threadIdx.x] = v;
        svidx[threadIdx.x] = v_pos;
      } else {
        sheight[threadIdx.x] = INF;
        svid[threadIdx.x] = -1;
        svidx[threadIdx.x] = -1;
      }
    } else {
      sheight[threadIdx.x] = INF;
      svid[threadIdx.x] = -1;
      svidx[threadIdx.x] = -1;
    }
    tile.sync();
    for (unsigned int s = tile.size() / 2; s > 0; s >>= 1) {
      if (idx < s) {
        if ((sheight[threadIdx.x] > sheight[threadIdx.x + s])) {
          sheight[threadIdx.x] = sheight[threadIdx.x + s];
          svid[threadIdx.x] = svid[threadIdx.x + s];
          svidx[threadIdx.x] = svidx[threadIdx.x + s];
        }
      }
      tile.sync();
    }
    tile.sync();
    if (idx == 0) {
      if (minH >
          sheight[threadIdx.x]) { // The address of the first thread in the tile
        minH = sheight[threadIdx.x];
        minV = svid[threadIdx.x];
        *v_index = svidx[threadIdx.x];
      }
    }
    tile.sync();
    svid[threadIdx.x] = -1;
    sheight[threadIdx.x] = INF;
    tile.sync();
  }
  tile.sync();
  return minV;
}

__global__ void generateDegreeDAG(deviceGraphPointers G, deviceDAGpointer D,
                                  ui *listingOrder, ui n, ui m, ui totalWarps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < n; i += totalWarps) {
    ui start = G.offset[i];
    ui end = G.offset[i + 1];
    ui total = end - start;
    ui neigh;
    int count = 0;
    for (int j = laneId; j < total; j += warpSize) {
      neigh = G.neighbors[start + j];
      if (listingOrder[i] < listingOrder[neigh]) {
        count++;
      }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (laneId == 0) {
      D.degree[i] = count;
    }
  }
}

__global__ void generateNeighborDAG(deviceGraphPointers G, deviceDAGpointer D,
                                    ui *listingOrder, ui n, ui m,
                                    ui totalWarps) {

  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;

  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < n; i += totalWarps) {
    if (laneId == 0) {
      counter[threadIdx.x / warpSize] = D.offset[i];
    }
    __syncwarp();
    ui start = G.offset[i];
    ui end = G.offset[i + 1];
    ui total = end - start;
    ui neigh;
    for (int j = laneId; j < total; j += warpSize) {
      neigh = G.neighbors[start + j];

      if (listingOrder[i] < listingOrder[neigh]) {
        int loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);
        D.neighbors[loc] = neigh;
      }
    }
    __syncwarp();
  }
}

__global__ void listIntialCliques(deviceDAGpointer D,
                                  cliqueLevelDataPointer levelData, ui *label,
                                  ui k, ui n, ui psize, ui cpSize,
                                  ui maxBitMask, ui level, ui totalWarps,
                                  size_t partialSize, size_t candidateSize,
                                  size_t maskSize, size_t offsetSize) {
  extern __shared__ char sharedMemory[];
  ui *counter = (ui *)sharedMemory;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  // Use size_t for all large calculations to prevent overflow
  size_t oP = (size_t)warpId * (size_t)psize;
  size_t oCP = (size_t)warpId * (size_t)cpSize;
  size_t oO = (size_t)warpId * ((size_t)psize / (k - 1) + 1);
  size_t oM = oCP * (size_t)maxBitMask;
  
  // Calculate word-based offset for 32-bit atomic operations
  size_t labelWordOffset = ((size_t)warpId * (size_t)n + 31) / 32;

  // Create pointers to warp-specific partitions
  ui *warpPartialCliques = levelData.partialCliquesPartition + oP;
  ui *warpCandidates = levelData.candidatesPartition + oCP;
  ui *warpOffsetPartition = levelData.offsetPartition + oO;
  ui *warpValidNeighMask = levelData.validNeighMaskPartition + oM;
  ui *warpCount = levelData.count + warpId + 1;
  ui *warpLabel = (ui *)label + labelWordOffset; // Pointer to this warp's label words

  for (ui i = warpId; i < n; i += totalWarps) {
    ui vertex = i, vOff = D.offset[vertex];

    if (laneId == 0)
      counter[threadIdx.x / warpSize] = 0;
    __syncwarp();

    // Use size_t for offset calculations
    ui cOff = warpOffsetPartition[*warpCount];

    for (ui j = laneId; j < D.degree[vertex]; j += warpSize) {
      ui neigh = D.neighbors[vOff + j];

      // 32-bit atomic bit manipulation
      ui wordIdx = neigh / 32;        // Which ui word (0, 1, 2, ...)
      ui bitPos = neigh % 32;         // Which bit in that word (0-31)
      ui mask = 1U << bitPos;         // Create mask for that bit

      ui old = atomicOr(&warpLabel[wordIdx], mask);
      if (!(old & mask)) {
        // This neighbor wasn't set before
        ui loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);
        warpCandidates[cOff + loc] = neigh;
      }
    }

    __syncwarp();

    if (laneId == 0 && counter[threadIdx.x / warpSize] > 0) {
      size_t w = (size_t)(*warpCount) * (k - 1) + level;
      warpPartialCliques[w] = vertex;

      *warpCount += 1;

      warpOffsetPartition[*warpCount] =
          warpOffsetPartition[*warpCount - 1] + counter[threadIdx.x / warpSize];
    }

    __syncwarp();

    ui counterValue = counter[threadIdx.x / warpSize];
    for (ui j = laneId; j < counterValue; j += warpSize) {
      ui cand = warpCandidates[cOff + j];
      ui dOff = D.offset[cand], deg = D.degree[cand];
      ui chunks = (deg + 31) / 32;

      for (ui m = 0; m < chunks; m++) {
        ui bitmask = 0, from = m * 32;
        ui to = min(from + 32, deg);

        for (ui x = from; x < to; x++) {
          ui nb = D.neighbors[dOff + x];

          // 32-bit atomic bit check
          ui wordIdx = nb / 32;
          ui bitPos = nb % 32;
          ui mask = 1U << bitPos;

          if ((warpLabel[wordIdx] & mask) != 0)
            bitmask |= 1 << (x - from);
        }

        // Use size_t for large index calculations
        size_t maskIdx = ((size_t)warpOffsetPartition[*warpCount - 1] + j) *
                             (size_t)maxBitMask +
                         m;
        warpValidNeighMask[maskIdx] = bitmask;
      }
    }

    __syncwarp();

    // Clear this warp's label bits using 32-bit atomic operations
    for (ui x = laneId; x < n; x += warpSize) {
      ui wordIdx = x / 32;
      ui bitPos = x % 32;
      ui mask = ~(1U << bitPos);    // Inverted mask to clear the bit
      atomicAnd(&warpLabel[wordIdx], mask);
    }

    __syncwarp();
  }
}

__global__ void flushParitions(deviceDAGpointer D,
                               cliqueLevelDataPointer levelData, ui pSize,
                               ui cpSize, ui k, ui maxBitMask, ui level,
                               ui totalWarps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;
  int cliquePartition = warpId * pSize;
  int offsetPartition = warpId * (pSize / (k - 1) + 1);
  int candidatePartition = warpId * cpSize;
  int maskPartition = warpId * cpSize * maxBitMask;

  int totalTasks = levelData.count[warpId + 1] - levelData.count[warpId];

  for (int iter = 0; iter < totalTasks; iter++) {
    int start =
        candidatePartition + levelData.offsetPartition[offsetPartition + iter];
    int end = candidatePartition +
              levelData.offsetPartition[offsetPartition + iter + 1];
    int total = end - start;
    int writeOffset = levelData.temp[warpId] +
                      levelData.offsetPartition[offsetPartition + iter];
    for (int i = laneId; i < total; i += warpSize) {
      ui candidate = levelData.candidatesPartition[start + i];
      levelData.candidates[writeOffset + i] = candidate;

      int totalMasks = (D.degree[candidate] + 31) / 32;
      for (int j = 0; j < totalMasks; j++) {
        levelData.validNeighMask[(writeOffset + i) * maxBitMask + j] =
            levelData.validNeighMaskPartition
                [maskPartition +
                 (levelData.offsetPartition[offsetPartition + iter] + i) *
                     maxBitMask +
                 j];
      }
    }

    if (laneId < level + 1) {

      levelData.partialCliques[levelData.count[warpId] * (k - 1) +
                               iter * (k - 1) + laneId] =
          levelData.partialCliquesPartition[cliquePartition + iter * (k - 1) +
                                            laneId];
    }

    __syncwarp();

    if (laneId == 0) {

      levelData.offset[levelData.count[warpId] + iter + 1] =
          levelData.temp[warpId] +
          levelData.offsetPartition[offsetPartition + iter + 1];
    }
    __syncwarp();
  }
}

__global__ void listMidCliques(deviceDAGpointer D,
                               cliqueLevelDataPointer levelData, ui *label,
                               ui k, ui iterK, ui n, ui m, ui pSize, ui cpSize,
                               ui maxBitMask, ui totalTasks, ui level,
                               ui totalWarps) {

  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;

  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;
  int cliquePartition = warpId * pSize;
  int offsetPartition = warpId * (pSize / (k - 1) + 1);
  int candidatePartition = warpId * cpSize;
  int maskPartition = warpId * cpSize * maxBitMask;

  // Iter through total pc.
  for (int i = warpId; i < totalTasks; i += totalWarps) {

    int start = levelData.offset[i];
    int totalCandidates = levelData.offset[i + 1] - start;
    // Process one canddate at a time
    for (int iter = 0; iter < totalCandidates; iter++) {
      int candidate = levelData.candidates[start + iter];
      if (laneId == 0) {
        counter[threadIdx.x / warpSize] = 0;
      }
      __syncwarp();

      int degree = D.degree[candidate];
      int neighOffset = D.offset[candidate];

      int writeOffset =
          candidatePartition +
          levelData
              .offsetPartition[offsetPartition + levelData.count[warpId + 1]];
      for (int j = laneId; j < degree; j += warpSize) {
        int iterBitMask = j / warpSize;
        int bitPos = j % 32;
        int neighBitMask =
            levelData.validNeighMask[(start + iter) * maxBitMask + iterBitMask];
        ui neigh = D.neighbors[neighOffset + j];
        if (neighBitMask & (1 << bitPos)) {
          size_t bitIndex = static_cast<size_t>(warpId) * n + neigh;
          size_t byteIdx = bitIndex / 8;
          uint8_t btMask = 1 << (bitIndex % 8);

          uint8_t oldByte = atomicOr(&label[byteIdx], btMask);
          //ui old_val = atomicCAS(&label[warpId * n + neigh], iterK, iterK - 1);
          if (!(oldByte & btMask)) {
            ui loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);
            levelData.candidatesPartition[writeOffset + loc] = neigh;
          }
        }
      }
      __syncwarp();
      if (laneId == 0 && counter[threadIdx.x / warpSize] > 0) {
        levelData.partialCliquesPartition
            [cliquePartition + levelData.count[warpId + 1] * (k - 1) + level] =
            candidate;
        for (int l = 0; l < level; l++) {
          levelData.partialCliquesPartition
              [cliquePartition + levelData.count[warpId + 1] * (k - 1) + l] =
              levelData.partialCliques[i * (k - 1) + l];
        }
        levelData.count[warpId + 1] += 1;
        levelData
            .offsetPartition[offsetPartition + levelData.count[warpId + 1]] =
            levelData.offsetPartition[offsetPartition +
                                      levelData.count[warpId + 1] - 1] +
            counter[threadIdx.x / warpSize];
      }

      __syncwarp();

      int start = writeOffset;

      for (int j = laneId; j < counter[threadIdx.x / warpSize]; j += warpSize) {
        int cand = levelData.candidatesPartition[start + j];
        int neighOffset = D.offset[cand];
        int degree = D.degree[cand];

        int numBitmasks = (degree + 31) / 32;

        for (int bitmaskIndex = 0; bitmaskIndex < numBitmasks; bitmaskIndex++) {
          ui bitmask = 0; // Initialize bitmask to 0

          // Iterate over the current chunk of 32 neighbors
          int startNeighbor = bitmaskIndex * 32;
          int endNeighbor = min(startNeighbor + 32, degree);
          for (int x = startNeighbor; x < endNeighbor; x++) {
            size_t bitIndex = static_cast<size_t>(warpId) * n + D.neighbors[neighOffset + x];
            size_t byteIdx = bitIndex / 8;
            uint8_t btMask = 1 << (bitIndex % 8);

            if ((label[byteIdx] & btMask) != 0) {
              bitmask |=
                  (1 << (x - startNeighbor)); // Set the bit for valid neighbors
            }
          }

          levelData.validNeighMaskPartition
              [maskPartition +
               (levelData.offsetPartition[offsetPartition +
                                          levelData.count[warpId + 1] - 1] +
                j) *
                   maxBitMask +
               bitmaskIndex] = bitmask;
        }
      }

      __syncwarp();

     for (int x = laneId; x < n; x += 32) {
      size_t bitIndex = static_cast<size_t>(warpId) * n + x;
      size_t byteIdx = bitIndex / 8;
      uint8_t btMask = ~(1 << (bitIndex % 8));  // inverted mask

      // Atomically clear the bit
      atomicAnd(&label[byteIdx], btMask);
    }


      __syncwarp();
    }
  }
}

__global__ void writeFinalCliques(deviceGraphPointers G, deviceDAGpointer D,
                                  cliqueLevelDataPointer levelData,
                                  deviceCliquesPointer cliqueData,
                                  ui *globalCounter, ui k, ui iterK, ui n, ui m,
                                  ui pSize, ui cpSize, ui maxBitMask,
                                  ui trieSize, ui totalTasks, ui level,
                                  ui totalWarps) {
  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;
  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (int i = warpId; i < totalTasks; i += totalWarps) {

    int start = levelData.offset[i];
    int totalCandidates = levelData.offset[i + 1] - start;

    for (int iter = 0; iter < totalCandidates; iter++) {
      int candidate = levelData.candidates[start + iter];
      if (laneId == 0) {
        counter[threadIdx.x / warpSize] = 0;
      }
      __syncwarp();
      int degree = D.degree[candidate];
      int neighOffset = D.offset[candidate];

      for (int j = laneId; j < degree; j += warpSize) {
        int iterBitMask = j / warpSize;
        int bitPos = j % 32;
        int neighBitMask =
            levelData.validNeighMask[(start + iter) * maxBitMask + iterBitMask];
        if (neighBitMask & (1 << bitPos)) {

          ui neigh = D.neighbors[neighOffset + j];

          ui loc = atomicAdd(globalCounter, 1);
          for (int ind = 0; ind < k - 2; ind++) {
            cliqueData.trie[trieSize * ind + loc] =
                levelData.partialCliques[(i) * (k - 1) + ind];
          }
          atomicAdd(&counter[threadIdx.x / warpSize], 1);
          cliqueData.trie[trieSize * (k - 2) + loc] = candidate;
          cliqueData.trie[trieSize * (k - 1) + loc] = neigh;
          cliqueData.status[loc] = -1;
          atomicAdd(&G.cliqueDegree[neigh], 1);
          atomicAdd(&G.cliqueDegree[candidate], 1);
        }
      }
      __syncwarp();

      for (int j = laneId; j < k - 2; j += warpSize) {
        int pClique = levelData.partialCliques[i * (k - 1) + j];
        atomicAdd(&G.cliqueDegree[pClique], counter[threadIdx.x / warpSize]);
      }
      __syncwarp();
    }
  }
}

__global__ void writeEdgeCliques(deviceGraphPointers G, deviceDAGpointer D,
                                 deviceCliquesPointer cliqueData,
                                 ui *cliqueCount, ui n, ui m, ui pSize,
                                 ui trieSize, ui totalWarps) {
  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;
  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (int i = warpId; i < n; i += totalWarps) {

    int start = D.offset[i];
    int total = D.offset[i + 1] - start;

    if (laneId == 0) {
      counter[threadIdx.x / warpSize] = start;
    }
    __syncwarp();

    for (int j = laneId; j < total; j += warpSize) {
      ui loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);

      cliqueData.trie[trieSize * 0 + loc] = i;
      cliqueData.trie[trieSize * 1 + loc] = D.neighbors[start + j];
      int core = min(G.cliqueCore[i], G.cliqueCore[D.neighbors[start + j]]);
      cliqueData.status[loc] = core;
      atomicAdd(&cliqueCount[core], 1);
    }
    __syncwarp();
  }
}

__global__ void selectNodes(deviceGraphPointers G, ui *bufTails, ui *glBuffers,
                            ui glBufferSize, ui n, ui level) {
  __shared__ ui *glBuffer;
  __shared__ ui bufTail;

  if (threadIdx.x == 0) {
    bufTail = 0;
    glBuffer = glBuffers + blockIdx.x * glBufferSize;
  }
  __syncthreads();
  ui total = (n + BLK_NUMS) / BLK_NUMS;
  ui start = blockIdx.x * total;
  ui end = min(start + total, n);

  ui thid = threadIdx.x;

  for (ui i = start + thid; i < end; i += BLK_DIM) {
    ui v = i;
    if (G.cliqueCore[v] == level) {
      ui loc = atomicAdd(&bufTail, 1);
      assert(loc < glBufferSize && "glBuffer overflow in selectNodes");
      glBuffer[loc] = v;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    assert(bufTail <= glBufferSize &&
           "bufTail exceeds glBufferSize in selectNodes");
    bufTails[blockIdx.x] = bufTail;
  }
}

__global__ void processNodesByWarp(deviceGraphPointers G,
                                   deviceCliquesPointer cliqueData,
                                   ui *bufTails, ui *glBuffers, ui *globalCount,
                                   ui glBufferSize, ui n, ui level, ui k,
                                   ui totalCliques) {
  /*removes the verties to get their core value.
    Warp processes the verticies in its virtual partition parallely*/
  __shared__ ui bufTail;
  __shared__ ui *glBuffer;
  __shared__ ui base;
  ui warpId = threadIdx.x / 32;
  ui laneId = threadIdx.x % 32;
  ui regTail;
  ui i;
  if (threadIdx.x == 0) {
    // index of last vertex in vitual partition of glguffer for current warp
    bufTail = bufTails[blockIdx.x];

    // stores index of current processed vertex
    base = 0;
    glBuffer = glBuffers + blockIdx.x * glBufferSize;
    assert(glBuffer != NULL);
  }

  while (true) {
    __syncthreads();
    if (base == bufTail)
      break; // all the threads will evaluate to true at same iteration

    i = base + warpId;
    regTail = bufTail;
    __syncthreads();

    if (i >= regTail)
      continue; // this warp won't have to do anything

    if (threadIdx.x == 0) {
      base += WARPS_EACH_BLK;
      if (regTail < base)
        ;
      base = regTail;
    }

    // vertex to be removed
    ui v = glBuffer[i];

    __syncwarp();

    // warp iters through the clique data.
    for (ui j = laneId; j < totalCliques; j += warpSize) {

      // if valid clique and not removed yet
      if (cliqueData.status[j] == -1) {

        // flag to check if vertex found in the clique
        bool found = false;
        // stores the index at which the vertex was found in clique (0,k-1)
        ui w = 0;

        // iter through verticies of clique sequentially.
        while (w < k) {

          if (cliqueData.trie[w * totalCliques + j] == v) {

            found = true;
          }
          w++;
        }

        if (found) {
          cliqueData.status[j] = level;
          // iter throught the clique verticies
          for (ui x = 0; x < k; x++) {

            // continue it clique vertex is same as vertex
            // clique vertex
            ui u = cliqueData.trie[x * totalCliques + j];
            if (u == v)
              continue;

            // decreament its core value by 1
            int a = atomicSub(&G.cliqueCore[u], 1);

            // if core value is less than level, update to level.
            if (a - 1 < level) {
              atomicExch(&G.cliqueCore[u], level);
            }
            // if core value is level, add to glbuffer so can be removed in this
            // level.
            if (a == level + 1) {
              ui loc = atomicAdd(&bufTail, 1);
              glBuffer[loc] = u;
            }
          }

          // set status of the clique to current core level.
        }
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(globalCount, bufTail); // atomic since contention among blocks
  }
}

__global__ void processNodesByBlock(deviceGraphPointers G,
                                    deviceCliquesPointer cliqueData,
                                    ui *bufTails, ui *glBuffers,
                                    ui *globalCount, ui glBufferSize, ui n,
                                    ui level, ui k, ui t, ui tt) {
  __shared__ ui bufTail;
  __shared__ ui *glBuffer;
  __shared__ ui base;

  ui regTail;
  ui i;
  if (threadIdx.x == 0) {
    bufTail = bufTails[blockIdx.x];
    base = 0;
    glBuffer = glBuffers + blockIdx.x * glBufferSize;
    assert(glBuffer != NULL);
  }

  while (true) {
    __syncthreads();
    if (base == bufTail)
      break; // all the threads will evaluate to true at same iteration
    i = base + blockIdx.x;
    regTail = bufTail;
    __syncthreads();

    if (i >= regTail)
      continue; // this warp won't have to do anything

    if (threadIdx.x == 0) {
      base += 1;
      if (regTail < base)
        base = regTail;
    }
    // bufTail is incremented in the code below:
    ui v = glBuffer[i];

    __syncthreads();
    ui idx = threadIdx.x;

    for (ui j = idx; j < tt; j += BLK_DIM) {

      if ((v == cliqueData.trie[j]) && (cliqueData.status[j] == -1)) {
        for (ui x = 1; x < k; x++) {
          ui u = cliqueData.trie[x * t + j];
          int a = atomicSub(&G.cliqueCore[u], 1);
          if (a == level + 1) {
            ui loc = atomicAdd(&bufTail, 1);
            glBuffer[loc] = u;
          }
          if (a <= level) {
            atomicAdd(&G.cliqueCore[u], 1);
          }
          if (G.cliqueCore[u] < 0) {
            G.cliqueCore[u] = 0;
          }
        }
        cliqueData.status[j] = level;
      }
    }

    __syncthreads();

    if (threadIdx.x == 0 && bufTail > 0) {
      atomicAdd(globalCount, 1); // atomic since contention among blocks
    }
  }
}

__global__ void generateDensestCore(deviceGraphPointers G,
                                    densestCorePointer densestCore,
                                    ui *globalCount, ui n, ui core,
                                    ui totalWarps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < n; i += totalWarps) {
    if (G.cliqueCore[i] >= core) {

      ui loc;
      if (laneId == 0) {
        loc = atomicAdd(globalCount, 1);
        densestCore.mapping[loc] = i;
      }
      loc = __shfl_sync(0xFFFFFFFF, loc, 0, 32);
      ui start = G.offset[i];
      ui end = G.offset[i + 1];
      ui total = end - start;
      ui neigh;
      int count = 0;
      for (int j = laneId; j < total; j += warpSize) {
        neigh = G.neighbors[start + j];
        if (G.cliqueCore[neigh] >= core) {
          count++;
        }
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);
      }
      __syncwarp();

      if (laneId == 0) {
        densestCore.offset[loc + 1] = count;
      }
      __syncwarp();
    }
  }
}

__global__ void generateNeighborDensestCore(deviceGraphPointers G,
                                            densestCorePointer densestCore,
                                            ui core, ui totalWarps) {

  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;

  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < (*densestCore.n); i += totalWarps) {
    if (laneId == 0) {
      counter[threadIdx.x / warpSize] = densestCore.offset[i];
    }
    __syncwarp();
    ui vertex = densestCore.mapping[i];
    ui start = G.offset[vertex];
    ui end = G.offset[vertex + 1];
    ui total = end - start;
    ui neigh;
    for (int j = laneId; j < total; j += warpSize) {
      neigh = G.neighbors[start + j];

      if (G.cliqueCore[neigh] >= core) {
        int loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);

        densestCore.neighbors[loc] = densestCore.reverseMap[neigh];
      }
    }
    __syncwarp();
  }
}

__global__ void pruneEdges(densestCorePointer densestCore,
                           deviceCliquesPointer cliqueData, ui *pruneStatus,
                           ui totalCliques, ui k, ui level) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < totalCliques; i += TOTAL_WARPS) {

    if (cliqueData.status[i] >= level) {

      for (ui iter = 0; iter < k; iter++) {
        // v should be mapped
        ui u_ = ((iter) % k) * totalCliques + i;
        ui u = densestCore.reverseMap[cliqueData.trie[u_]];
        for (ui j = 0; j < k; j++) {
          ui v_ = ((j) % k) * totalCliques + i;

          if (v_ != u_) {
            int v = densestCore.reverseMap[cliqueData.trie[v_]];

            // Update u-v edge status
            ui start = densestCore.offset[u];
            ui end = densestCore.offset[u + 1];
            ui total = end - start;

            for (ui ind = laneId; ind < total; ind += WARPSIZE) {
              int neigh = densestCore.neighbors[start + ind];
              if (neigh == v) {
                atomicCAS(&pruneStatus[start + ind], 1, 0);
              }
            }
          }
        }
      }
    }
  }
}

__global__ void componentDecomposek(deviceComponentPointers conComp,
                                    devicePrunedNeighbors prunedNeighbors,
                                    ui *changed, ui n, ui m, ui totalWarps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;
  bool threadChanged = false;

  for (ui i = warpId; i < n; i += totalWarps) {
    ui currentComp = conComp.components[i];
    ui start = prunedNeighbors.newOffset[i];
    ui end = prunedNeighbors.newOffset[i + 1];
    ui total = end - start;

    ui minNeighComp = currentComp;

    for (ui j = laneId; j < total; j += warpSize) {
      ui neighComp =
          conComp.components[prunedNeighbors.newNeighbors[start + j]];
      minNeighComp = min(minNeighComp, neighComp);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      ui temp = __shfl_down_sync(0xFFFFFFFF, minNeighComp, offset);
      minNeighComp = min(minNeighComp, temp);
    }

    if (laneId == 0) {
      if (minNeighComp < currentComp) {
        conComp.components[i] = minNeighComp;
        threadChanged = true;
      }
    }

    __syncwarp();
  }

  bool warpChanged = __any_sync(0xFFFFFFFF, threadChanged);
  if (warpChanged && laneId == 0) {
    atomicAdd(changed, 1);
  }
}

__global__ void generateDegreeAfterPrune(densestCorePointer densestCore,
                                         ui *pruneStatus, ui *newOffset, ui n,
                                         ui m, ui totalWarps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < n; i += totalWarps) {
    ui start = densestCore.offset[i];
    ui end = densestCore.offset[i + 1];
    ui total = end - start;
    int count = 0;
    for (int j = laneId; j < total; j += warpSize) {
      if (!pruneStatus[start + j]) {
        count++;
      }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (laneId == 0) {
      newOffset[i + 1] = count;
    }
  }
}

__global__ void generateNeighborAfterPrune(densestCorePointer densestCore,
                                           ui *pruneStatus, ui *newOffset,
                                           ui *newNeighbors, ui n, ui m,
                                           ui totalWarps) {

  extern __shared__ char sharedMemory[];
  ui sizeOffset = 0;

  ui *counter = (ui *)(sharedMemory + sizeOffset);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < n; i += totalWarps) {
    if (laneId == 0) {
      counter[threadIdx.x / warpSize] = newOffset[i];
    }
    __syncwarp();
    ui start = densestCore.offset[i];
    ui end = densestCore.offset[i + 1];
    ui total = end - start;
    ui neigh;
    for (int j = laneId; j < total; j += warpSize) {
      neigh = densestCore.neighbors[start + j];

      if (!pruneStatus[start + j]) {
        int loc = atomicAdd(&counter[threadIdx.x / warpSize], 1);
        newNeighbors[loc] = neigh;
      }
    }
    __syncwarp();
  }
}

__global__ void getConnectedComponentStatus(deviceComponentPointers conComp,
                                            deviceCliquesPointer cliqueData,
                                            densestCorePointer densestCore,
                                            ui *compCounter, ui totalCliques,
                                            ui k, ui maxCore, ui totalThreads) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (ui i = idx; i < totalCliques; i += totalThreads) {
    if (cliqueData.status[i] >= maxCore) {
      int comp = INT_MAX;

      for (ui x = 0; x < k; x++) {
        ui vertex =
            densestCore.reverseMap[cliqueData.trie[x * totalCliques + i]];
        comp = min(comp, conComp.components[conComp.reverseMapping[vertex]]);

        cliqueData.trie[x * totalCliques + i] = vertex;
      }
      cliqueData.status[i] = comp;
      atomicAdd(&compCounter[comp + 1], 1);

    } else {
      cliqueData.status[i] = -1;
    }
  }
}

__global__ void rearrangeCliqueData(deviceComponentPointers conComp,
                                    deviceCliquesPointer cliqueData,
                                    deviceCliquesPointer finalCliqueData,
                                    densestCorePointer densestCore,
                                    ui *compCounter, ui *counter,
                                    ui totaLCliques, ui k, ui newTotaLCliques,
                                    ui totalThreads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (ui i = idx; i < totaLCliques; i += totalThreads) {

    int comp = cliqueData.status[i];

    if (comp > -1) {
      ui loc;
      for (ui j = 0; j < k; j++) {
        ui vertex = cliqueData.trie[j * totaLCliques + i];
        ui offset = compCounter[comp];
        if (j == 0) {
          loc = atomicAdd(&counter[comp], 1);
          finalCliqueData.status[offset + loc] = comp;
        }
        finalCliqueData.trie[offset + j * newTotaLCliques + loc] = vertex;
      }
    }
  }
}

__global__ void countCliques(deviceDAGpointer D,
                             cliqueLevelDataPointer levelData,
                             ui *globalCounter, ui maxBitMask, ui totalTasks,
                             ui totalWarps) {
  /* Find Total number of cliques in the graph by counting the valid neighbors.
     Each warp processes on partial clique. */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (int i = warpId; i < totalTasks; i += totalWarps) {

    // candidate offset
    int start = levelData.offset[i];
    int totalCandidates = levelData.offset[i + 1] - start;

    int count = 0;

    for (int j = laneId; j < totalCandidates; j += warpSize) {
      int degree = D.degree[levelData.candidates[start + j]];
      int numBitmasks = (degree + 31) / 32;
      for (int x = 0; x < numBitmasks; x++) {
        int neighBitMask =
            levelData.validNeighMask[(start + j) * maxBitMask + x];
        count += __popc(neighBitMask);
      }
    }

    __syncwarp();

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (laneId == 0) {
      atomicAdd(globalCounter, count);
    }
  }
}

__global__ void getLbUbandSize(deviceComponentPointers conComp, ui *compCounter,
                               double *lowerBound, double *upperBound,
                               ui *ccOffset, ui *neighborSize,
                               ui totalComponenets, ui k, double maxDensity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    ccOffset[idx] = 0;
    neighborSize[idx] = 0;
  }

  for (ui i = idx; i < totalComponenets; i += TOTAL_THREAD) {
    ui totalCliques = compCounter[i + 1] - compCounter[i];
    ui totalSize = conComp.componentOffset[i + 1] - conComp.componentOffset[i];
    double lb = (double)(totalCliques) / totalSize;
    lowerBound[i] = lb;

    double dem = pow(fact(k), 1.0 / k);
    double num = pow(totalCliques, (k - 1.0) / k);
    double ub = min(maxDensity, num / dem);

    upperBound[i] = ub;

    if (ub > lb) {
      ccOffset[i] = totalCliques + totalSize + 2;
      neighborSize[i] = (2 * totalCliques * k + 4 * totalSize);

    } else {
      ccOffset[i + 1] = 0;
      neighborSize[i + 1] = 0;
    }
  }
}

__global__ void createFlowNetworkOffset(deviceFlowNetworkPointers flowNetwork,
                                        deviceComponentPointers conComp,
                                        deviceCliquesPointer finalCliqueData,
                                        ui *compCounter, ui iter, ui k, ui t) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  ui startClique = compCounter[iter];
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];

  // offset for verticies
  for (ui i = idx; i < totalCliques; i += TOTAL_THREAD) {
    for (ui x = 0; x < k; x++) {
      ui u =
          conComp
              .reverseMapping[finalCliqueData.trie[t * x + startClique + i]] -
          start;
      atomicAdd(&flowNetwork.offset[u + 1], 1);
    }
  }

  // offset for cliques
  for (ui j = idx; j < totalCliques; j += TOTAL_THREAD) {
    flowNetwork.offset[total + j + 1] = k;
  }
  // offset for source and sink
  if (idx == 0) {
    flowNetwork.offset[total + totalCliques + 1] = total;
    flowNetwork.offset[total + totalCliques + 2] = total;
    flowNetwork.offset[0] = 0;
  }
}

__global__ void createFlowNetwork(deviceFlowNetworkPointers flowNetwork,
                                  deviceComponentPointers conComp,
                                  deviceCliquesPointer finalCliqueData,
                                  ui *compCounter, double *upperBound,
                                  double *lowerBound, ui *counter, ui iter,
                                  ui k, ui t) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  ui startClique = compCounter[iter];
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];
  double alpha = (upperBound[iter] + lowerBound[iter]) / 2;
  for (ui i = idx; i < total; i += TOTAL_THREAD) {

    // vertex to sink forward -- (1)
    ui foffset = flowNetwork.offset[i];
    flowNetwork.neighbors[foffset] = total + totalCliques + 1;
    flowNetwork.capacity[foffset] = alpha * k;
    flowNetwork.flow[foffset] = alpha * k;
    // index of backward edge
    flowNetwork.flowIndex[foffset] =
        flowNetwork.offset[total + totalCliques + 1] + i;

    // source to vertex forward -- (2)
    ui foffset1 = flowNetwork.offset[total + totalCliques];
    flowNetwork.neighbors[foffset1 + i] = i;
    ui cliqueDegree = flowNetwork.offset[i + 1] - flowNetwork.offset[i] - 2;
    flowNetwork.capacity[foffset1 + i] = cliqueDegree;
    flowNetwork.flow[foffset1 + i] = cliqueDegree;

    flowNetwork.flowIndex[foffset1 + i] = flowNetwork.offset[i + 1] - 1;

    // sink to vertex backward of (1)
    ui foffset2 = flowNetwork.offset[total + totalCliques + 1];
    flowNetwork.neighbors[foffset2 + i] = i;
    flowNetwork.capacity[foffset2 + i] = 0;
    flowNetwork.flow[foffset2 + i] = 0;
    // index of forward edge --(1)
    flowNetwork.flowIndex[foffset2 + i] = flowNetwork.offset[i];

    // vertex to source backward of (2)
    ui foffset3 = flowNetwork.offset[i + 1] - 1;
    flowNetwork.neighbors[foffset3] = total + totalCliques;
    flowNetwork.capacity[foffset3] = 0;
    flowNetwork.flow[foffset3] = 0;
    // index to forward edge
    flowNetwork.flowIndex[foffset3] =
        flowNetwork.offset[total + totalCliques] + i;
  }

  for (ui i = idx; i < totalCliques; i += TOTAL_THREAD) {
    for (ui j = 0; j < k; j++) {
      ui u =
          conComp
              .reverseMapping[finalCliqueData.trie[t * j + startClique + i]] -
          start;

      // vertex to clique forward -- (1)
      ui foffset = flowNetwork.offset[u];
      ui loc = atomicAdd(&counter[u], 1);
      flowNetwork.neighbors[foffset + loc + 1] = total + i;
      flowNetwork.capacity[foffset + loc + 1] = 1;
      flowNetwork.flow[foffset + loc + 1] = 1;

      // index of backward clique to vertex
      flowNetwork.flowIndex[foffset + loc + 1] =
          flowNetwork.offset[total + i] + j;

      // clique to vertex backward -- (2)
      ui foffset1 = flowNetwork.offset[total + i];
      flowNetwork.neighbors[foffset1 + j] = u;
      flowNetwork.capacity[foffset1 + j] = k - 1;
      flowNetwork.flow[foffset1 + j] = k - 1;
      // index to foward
      flowNetwork.flowIndex[foffset1 + j] = foffset + loc + 1;
    }
  }
}

__global__ void preFlow(deviceFlowNetworkPointers flowNetwork,
                        deviceComponentPointers conComp, ui *compCounter,
                        double *totalExcess, ui iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];

  ui source = total + totalCliques;
  for (ui i = idx; i < (total + totalCliques + 2); i += TOTAL_THREAD) {
    flowNetwork.height[i] = (i == source) ? (total + totalCliques + 2) : 0;
    flowNetwork.excess[i] = 0;
  }

  ui nStart = flowNetwork.offset[total + totalCliques];

  for (ui i = idx; i < total; i += TOTAL_THREAD) {
    ui neigh = flowNetwork.neighbors[nStart + i];
    if (flowNetwork.capacity[nStart + i] > 1e-6) {
      flowNetwork.flow[nStart + i] = 0.0;
      ui backIndex = flowNetwork.flowIndex[nStart + i];
      flowNetwork.flow[backIndex] = flowNetwork.capacity[nStart + i];
      flowNetwork.excess[neigh] += flowNetwork.capacity[nStart + i];
      atomicAdd(totalExcess, flowNetwork.capacity[nStart + i]);
    }
  }
}

__global__ void pushRelabel(deviceFlowNetworkPointers flowNetwork,
                            deviceComponentPointers conComp, ui *compCounter,
                            double *totalExcess, ui *activeNodes, ui iter) {

  grid_group grid = this_grid();
  cg::thread_block block = cg::this_thread_block();
  const int tileSize = 32;
  cg::thread_block_tile<tileSize> tile = cg::tiled_partition<tileSize>(block);
  int numTilesPerBlock = (blockDim.x + tileSize - 1) / tileSize;
  int numTilesPerGrid = numTilesPerBlock * gridDim.x;
  int tileIdx = blockIdx.x * numTilesPerBlock + block.thread_rank() / tileSize;

  int minV = -1;
  // bool vinReverse = false;
  int v_index = -1;
  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  // ui startClique = compCounter[iter];
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];
  int totalFlow = total + totalCliques + 2;
  int cycle = totalFlow;
  extern __shared__ int SharedMemory[];
  int *sheight = SharedMemory;
  int *svid = (int *)&SharedMemory[blockDim.x];
  int *svidx = (int *)&svid[blockDim.x];
  ui source = totalFlow - 2;
  ui sink = totalFlow - 1;

  while (cycle > 0) {
    scan_active_vertices(totalFlow, source, sink, flowNetwork, activeNodes);

    grid.sync();

    if (globalCounter == 0) {
      break;
    }
    v_index = -1;
    grid.sync();

    for (int i = tileIdx; i < globalCounter; i += numTilesPerGrid) {
      ui u = activeNodes[i];

      minV = tiled_search_neighbor<tileSize>(tile, i, sheight, svid, svidx,
                                             &v_index, totalFlow, source, sink,
                                             flowNetwork, activeNodes);

      tile.sync();
      if (tile.thread_rank() == 0) {

        if (minV == -1) {
          flowNetwork.height[u] = totalFlow;
        } else {
          if (flowNetwork.height[u] > flowNetwork.height[minV]) {
            double d;
            if (flowNetwork.excess[u] > flowNetwork.flow[v_index]) {
              d = flowNetwork.flow[v_index];
            } else {
              d = flowNetwork.excess[u];
            }
            atomicAdd(&flowNetwork.flow[v_index], -d);
            atomicAdd(&flowNetwork.flow[flowNetwork.flowIndex[v_index]], d);

            atomicAdd(&flowNetwork.excess[minV], d);
            atomicAdd(&flowNetwork.excess[u], -d);
          } else {
            flowNetwork.height[u] = flowNetwork.height[minV] + 1;
          }
        }
      }
      tile.sync();
    }
    grid.sync();
    cycle = cycle - 1;
  }
}

__global__ void globalRelabel(deviceFlowNetworkPointers flowNetwork,
                              deviceComponentPointers conComp, ui *compCounter,
                              ui *changes, ui k, ui iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();

  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];

  for (ui i = idx; i < total; i += TOTAL_THREAD) {
    ui vertexOffset = flowNetwork.offset[i];
    if (flowNetwork.flow[vertexOffset] > 1e-8) {
      flowNetwork.height[i] = 1;

    }
  }

  grid.sync();

  for (ui i = idx; i < totalCliques; i += TOTAL_THREAD) {
    ui cliqueOffset = flowNetwork.offset[total + i];
    // ui v = flowNetwork.neighbors[cliqueOffset];
    for (ui j = 0; j < k; j++) {
      if (flowNetwork.flow[cliqueOffset + j] > 1e-8) {
        if (flowNetwork.height[flowNetwork.neighbors[cliqueOffset + j]] == 1) {
          flowNetwork.height[total + i] = 2;
          break;
        }
      }
    }
   
  }
  grid.sync();
  for (ui i = idx; i < total; i += TOTAL_THREAD) {
    ui sourceOffset = flowNetwork.offset[total + totalCliques];
    if (flowNetwork.flow[sourceOffset + i] > 1e-8) {
      ui v = flowNetwork.neighbors[sourceOffset + i];
      if (flowNetwork.height[v] == 1) {
        atomicCAS(changes, 0, 1);
      }
    }

  }
  grid.sync();
  if (idx == 0) {
    flowNetwork.height[total + totalCliques - 1] = 0;
    if (*changes == 1) {
      flowNetwork.height[total + totalCliques] = 2;
    }
  }
}

__global__ void updateFlownetwork(deviceFlowNetworkPointers flowNetwork,
                                  deviceComponentPointers conComp,
                                  deviceCliquesPointer finalCliqueData,
                                  ui *compCounter, double *upperBound,
                                  double *lowerBound, ui *gpuConverged,
                                  ui *gpuSize, double *gpuMaxDensity, ui k, ui t,
                                  ui iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();

  ui start = conComp.componentOffset[iter];
  ui end = conComp.componentOffset[iter + 1];
  ui total = end - start;
  ui cliqueStart = compCounter[iter];
  ui totalCliques = compCounter[iter + 1] - compCounter[iter];
  ui tFlow = total + totalCliques + 2;

  double bais = 1.0 / (total * (total - 1));
  double temp = static_cast<double>(totalCliques) * k;

  if (fabs(flowNetwork.excess[tFlow - 1] - temp) > 1e-8) {
    ui sourceOffset = flowNetwork.offset[total + totalCliques];
    for (ui i = idx; i < total; i += TOTAL_THREAD) {
      if (flowNetwork.flow[sourceOffset + i] > 1e-8) {
        atomicAdd(gpuSize, 1);
      }
    }

    grid.sync();

    for (ui i = idx; i < totalCliques; i += TOTAL_THREAD) {
      ui j = 0;
      ui found = true;
      while (j < k) {
        ui vertex =
            conComp
                .reverseMapping[finalCliqueData.trie[j * t + i + cliqueStart]] -
            start;

        if (flowNetwork.flow[sourceOffset + vertex] <= 1e-8) {
          found = false;
          break;
        }
        j++;
      }

      if (found) {
        atomicAdd(gpuMaxDensity, 1);
      }
    }
    grid.sync();
    if (idx == 0) {
      *gpuMaxDensity = *gpuMaxDensity / (*gpuSize);
    }
  } else {
    if (idx == 0) {
      *gpuMaxDensity = lowerBound[iter];
      //printf("max density %f iter %u lb %f \n", *gpuMaxDensity,iter,lowerBound[iter]);
      *gpuSize = total;
    }
  }
  grid.sync();
  double alpha = (upperBound[iter] + lowerBound[iter]) / 2;

  if (idx == 0) {
    if (fabs(flowNetwork.excess[tFlow - 1] - temp) < 1e-8) {
      upperBound[iter] = alpha;
    } else {
      lowerBound[iter] = alpha;
    }

    if ((upperBound[iter] - lowerBound[iter]) < bais) {
      *gpuConverged = 1;
    }
  }
  grid.sync();

  if (!*gpuConverged) {

    for (ui i = idx; i < total; i += TOTAL_THREAD) {
      ui vertexOffset = flowNetwork.neighbors[i];
      flowNetwork.capacity[vertexOffset] = alpha * k;
    }
  }
}