#include <catch2/catch_test_macros.hpp>
#include <pcg-cpp-0.98/include/pcg_random.hpp>
#include "Instruments.h"
#include "FinancialModels.h"
#include "MCLib.h"
#include "RNGs.h"
#include <algorithm>
#include <functional>


TEST_CASE("MersenneTwist RNG basic operations", "[RNG]") {
  MersenneTwistRNG rng;
  rng.initialize(10);

  std::vector<double> gaussian_vector(10);
  rng.get_gaussians(gaussian_vector);
  std::vector<double> antithetic{gaussian_vector};
  rng.get_gaussians(gaussian_vector);

  std::transform(gaussian_vector.begin(), gaussian_vector.end(),
                 antithetic.begin(), gaussian_vector.begin(),
                 std::plus<double>());


  auto it = std::find_if(gaussian_vector.begin(), gaussian_vector.end(),
                         [](auto x) { return x != 0.0; });

  // this tests the antithetic sampling
  REQUIRE(it == gaussian_vector.end());

  // now we test the jump ahead
  MersenneTwistRNG rng2;
  rng2.initialize(10);
  rng2.jump_ahead(1000);
  std::vector<double> gaussian_vector2(10);

  MersenneTwistRNG rng3;
  rng3.initialize(10);
  std::vector<double> gaussian_vector3(10);
  for(int i = 0; i < 1000; ++i) rng3.get_gaussians(gaussian_vector3);


  // run rng 3 until it catches up


  rng2.get_gaussians(gaussian_vector2);
  rng3.get_gaussians(gaussian_vector3);
  REQUIRE(gaussian_vector2 == gaussian_vector3);

}

TEST_CASE("PCG RNG basic operations", "[RNG]"){
  PCGRNG rng;
  rng.initialize(10);
  std::vector<double> gaussian_vector(10);
  rng.get_gaussians(gaussian_vector);
  std::vector<double> antithetic{gaussian_vector};
  rng.get_gaussians(gaussian_vector);

  std::transform(gaussian_vector.begin(), gaussian_vector.end(),
                 antithetic.begin(), gaussian_vector.begin(),
                 std::plus<double>());

  auto it = std::find_if(gaussian_vector.begin(), gaussian_vector.end(),
                         [](auto x) { return x != 0.0; });

  // This tests the antithetic sampling.
  REQUIRE(it == gaussian_vector.end());
}

TEST_CASE("European Call price", "[Instrument]"){
    // first test a call option that expires atm
    EuropeanCall<double> call{100.0, 1.0};
    std::vector<double> result(1, 0.0);
    Scenario<double> path(1);
    initialize_path(call.samples_needed(), path);
    call.payoffs(path, result);

    REQUIRE(result[0] == 0.0);


    EuropeanCall<double> call2{90.0, 0.0};
    call2.payoffs(path, result);
    REQUIRE(result[0] == 10.0);

    EuropeanCall<double> call3(110.0, 0.0);
    call3.payoffs(path, result);
    REQUIRE(result[0] == 0.0);
}


TEST_CASE("Black Scholes Model", "[FinancialModel]"){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  MersenneTwistRNG rng;

  auto cmodel = model.clone();

  cmodel -> attune(call);

  REQUIRE(cmodel -> simulation_dimension() == 1);


  std::vector<double> fake_gauss(cmodel->simulation_dimension(), 1.0);

  Scenario<double> path;
  initialize_path(call.samples_needed(), path);


  std::vector<std::vector<double>> result(1, std::vector<double>(1,0.0));
  cmodel->generate_path(fake_gauss, path);
  call.payoffs(path, result[0]);

  REQUIRE(std::abs(result[0][0] - 20.0) <= 0.5);
}

TEST_CASE("Simulation works!", "[Simulation]"){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  MersenneTwistRNG rng;

  auto result = monte_carlo_simulation(call, model, rng, 1000000);
  auto mean = std::accumulate(result.begin(), result.end(), 0.0l,
               [](auto sum, auto v){return sum + v[0] / 1000000;});
  REQUIRE(std::abs(mean - 7.965) <= 0.2);
}


TEST_CASE("Parallel Simulation Works!", "[Simulation]"){
  BlackScholesModel<double> model{100.0, 0.2};
  EuropeanCall<double> call{100.0, 1.0};
  PCGRNG rng;

  auto pool = ThreadPool::getInstance();
  pool->start();

  auto result = parallel_monte_carlo_simulation(call, model, rng, 1000000);
  auto price = std::accumulate(result.begin(), result.end(), 0.0l,
               [](auto acc, auto v){return acc + v[0];}) / 1000000;
  REQUIRE(std::abs(price - 7.965) <= 0.2);

  pool->stop();
}


TEST_CASE("up and out call", "[Instrument]"){
  BlackScholesModel<double> model{110.0, 0.1, 0.05};
  UpAndOutCall<double> exotic_option{100, 1.0, 140, 0.01, 0.01};
  PCGRNG rng;


  auto result = monte_carlo_simulation(exotic_option, model, rng, 100000);
  auto mean = std::accumulate(result.begin(), result.end(), 0.0l,
               [](auto sum, auto v){return sum + v[0] + v[1];}) / 100000;

  for(auto i = 0; i < 10; ++i){
    std::cout << result[i][0] << " " << result[i][1] << "\n";
  }
}