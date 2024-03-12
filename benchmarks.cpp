#include <benchmark/benchmark.h>
#include "MCLib.h"
#include "RNGs.h"



static void BM_Mersenne(benchmark::State& state) {
  MersenneTwistRNG rng;
  std::vector<double> gaussian_vector(100000);
  rng.initialize(gaussian_vector.size());
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
}
// Register the function as a benchmark
BENCHMARK(BM_Mersenne);

// Define another benchmark
static void BM_PCG(benchmark::State& state) {
  PCGRNG rng;
  std::vector<double> gaussian_vector(100000);
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
}
BENCHMARK(BM_PCG);

BENCHMARK_MAIN();