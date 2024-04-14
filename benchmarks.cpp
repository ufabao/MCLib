#include <benchmark/benchmark.h>
#include "FinancialModels.h"
#include "Instruments.h"
#include "MCLib.h"
#include "RNGs.h"
#include <list>




static void crazy(benchmark::State &state) {
  std::vector<int> v(100, 0);

  for (auto _ : state) {
    for (int i = 0; i < 100; ++i) {
      std::jthread thr([&v, i] { v[i] = 1; });
    }
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }
}

static void lesscrazy(benchmark::State &state) {
  std::vector<int> v(100, 0);

  for (auto _ : state) {
    for (int i = 0; i < 5; ++i) {
      std::jthread thr([&v, i] {
        for (int k = 0; k < 20; ++k) {
          v[(i % 4) * 20 + k] = 1;
        }
      });
    }
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }
}


static void normal(benchmark::State &state) {
  std::vector<int> v(100, 0);

  for (auto _ : state) {
    for (int i = 0; i < 100; ++i) {
      v[i] = 1;
    }
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }
}

static void parvv(benchmark::State &state) {
  std::vector<int> v(100, 2);

  std::vector<std::vector<int>> w(100, std::vector<int>(256, 0));

  for (auto _ : state) {
    for (auto i = 0; i < 100; ++i) {
      std::jthread th([&w, &v, i] { w[i] = v; });
    }

    benchmark::DoNotOptimize(w);
    benchmark::ClobberMemory();
  }
}

static void betterparvv(benchmark::State &state) {
  std::vector<int> v(100, 2);
  std::vector<std::vector<int>> w(100, std::vector<int>(256, 0));

  for (auto _ : state) {
    for (auto i = 0; i < 5; ++i) {
      std::jthread thr([&v, &w, i] {
        for (int k = 0; k < 5; ++k)
          w[(i % 5) * 20 + k] = v;
      });
    }
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }
}

static void stvv(benchmark::State &state) {
  std::vector<int> v(100, 2);
  std::vector<std::vector<int>> w(100, std::vector<int>(256, 0));

  for (auto _ : state) {
    for (auto i = 0; i < 100; ++i) {
      w[i] = v;
    }
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }
}

static void parlist(benchmark::State &state) {
  std::list<std::vector<int>> ls;
  for (auto i = 0; i < 100; ++i) {
    ls.emplace_back(std::vector<int>(256, 2));
  }

  const std::vector<int> v(256, 10);

  for (auto _ : state) {
    for (auto &w : ls) {
      std::jthread thr([&w, &v] { w = v; });
    }
    benchmark::DoNotOptimize(ls);
    benchmark::ClobberMemory();
  }
}

static void threadpool(benchmark::State &state) {
  auto pool = ThreadPool::getInstance();
  pool->start();

  std::vector<std::vector<int>> v(100, std::vector<int>(256, 2));
  std::vector<int> w(256, 10);
  vector<future<bool>> futures;
  futures.reserve(100);

  for (auto _ : state) {
    for (int i = 0; i < 100; ++i) {
      futures.push_back(pool->spawnTask([&, i]() {
        v[i] = w;
        return true;
      }));
    }
    for (auto &fut : futures)
      pool->activeWait(fut);
    benchmark::DoNotOptimize(v);
    benchmark::ClobberMemory();
  }

  pool->stop();
}


static void Mersenne(benchmark::State& state) {
  MersenneTwistRNG rng;
  std::vector<double> gaussian_vector(100000);
  rng.initialize(gaussian_vector.size());
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
    benchmark::DoNotOptimize(gaussian_vector);
    benchmark::ClobberMemory();
}



// Define another benchmark
static void PCG(benchmark::State& state) {
  PCGRNG rng;
  std::vector<double> gaussian_vector(100000);
  for (auto _ : state)
    rng.get_gaussians(gaussian_vector);
    benchmark::DoNotOptimize(gaussian_vector);
    benchmark::ClobberMemory();
}



static void parallel(benchmark::State& state){
  auto pool = ThreadPool::getInstance();
  //pool->start();
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  UpAndOutCall<double> exotic_option{110, 1.0, 130, 0.01, 0.01};

  PCGRNG rng;

  for(auto _ : state){  
    pool->start();
    auto result = parallel_monte_carlo_simulation(call, model, rng, 100000);
    auto price = std::accumulate(result.begin(), result.end(), 0.0l,
                 [](auto acc, auto v){return acc + v[0];}) / 100000;
    pool->stop();
    benchmark::DoNotOptimize(price);
    benchmark::ClobberMemory();
  }
}




static void serial(benchmark::State& state){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  UpAndOutCall<double> exotic_option{110, 1.0, 130, 0.01, 0.01};

  PCGRNG rng;

  for(auto _ : state){
    auto result = monte_carlo_simulation(call, model, rng, 100000);
    auto mean = std::accumulate(result.begin(), result.end(), 0.0l,
               [](auto sum, auto v){return sum + v[0] / 1000000;});
    benchmark::DoNotOptimize(mean);
    benchmark::ClobberMemory();
  }
}







//BENCHMARK(crazy);
//BENCHMARK(lesscrazy);
//BENCHMARK(normal);
//BENCHMARK(parvv);
//BENCHMARK(betterparvv);
//BENCHMARK(stvv);
//BENCHMARK(parlist);
//BENCHMARK(threadpool);


//BENCHMARK(Mersenne);
//BENCHMARK(PCG);

BENCHMARK(parallel);
BENCHMARK(serial);



BENCHMARK_MAIN();