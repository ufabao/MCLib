#include <benchmark/benchmark.h>
#include "FinancialModels.h"
#include "Instruments.h"
#include "MCLib.h"
#include "RNGs.h"



static void BM_Mersenne(benchmark::State& state) {
  MersenneTwistRNG rng;
  std::vector<double> gaussian_vector(100000);
  rng.initialize(gaussian_vector.size());
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
}



// Define another benchmark
static void BM_PCG(benchmark::State& state) {
  PCGRNG rng;
  std::vector<double> gaussian_vector(100000);
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
}



static void BM_parallel(benchmark::State& state){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  PCGRNG rng;

  for(auto _ : state){  
    auto result = parallel_monte_carlo_simulation(call, model, rng, 100000);
    auto price = std::accumulate(result.begin(), result.end(), 0.0l,
                 [](auto acc, auto v){return acc + v[0];}) / 100000;
    benchmark::DoNotOptimize(price);
    benchmark::ClobberMemory();
  }
}




static void BM_static(benchmark::State& state){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  MersenneTwistRNG rng;

  for(auto _ : state){
    auto result = monte_carlo_simulation(call, model, rng, 100000);
    auto mean = std::accumulate(result.begin(), result.end(), 0.0l,
               [](auto sum, auto v){return sum + v[0] / 1000000;});
    benchmark::DoNotOptimize(mean);
    benchmark::ClobberMemory();
  }
}











//BENCHMARK(BM_Mersenne);
//BENCHMARK(BM_PCG);


BENCHMARK(BM_parallel);
BENCHMARK(BM_static);



BENCHMARK_MAIN();