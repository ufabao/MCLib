#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include "newthreadpool.h"

template <typename T> 
struct SampleDef {
  bool numeraire = true;
  std::vector<double> forward_maturities;
  std::vector<double> discount_maturities;
};

template <typename T> 
struct MarketSample {
  T numeraire{T(1.0)};
  std::vector<T> forwards;
  std::vector<T> discounts;

  void initialize(const SampleDef<T> &data) {
    forwards.resize(data.forward_maturities.size());
    std::ranges::fill(forwards, T(100.0));

    discounts.resize(data.discount_maturities.size());
    std::ranges::fill(discounts, T(1.0));
  }
};

// We will typically be interested in vectors of MarketSamples to give us the
// flexibility to price exotic path dependent options
template <typename T> using Scenario = std::vector<MarketSample<T>>;

// Seperate allocation and initialization because allocation requires hidden
// locks, and for maximum performance we should be locked for the least amount
// of time possible
template <typename T>
inline void initialize_path(const std::vector<SampleDef<T>> &samples_needed,
                          Scenario<T>& path) {
  path.resize(samples_needed.size());
  for (auto i = 0; i < samples_needed.size(); ++i) {
    path[i].initialize(samples_needed[i]);
  }
}

// ABC interface for instruments. For our current purposes an instrument is
// mainly an exotic option, and mainly needs to support a function to compute
// its payoff given a simulated market scenario. The product also needs to be
// able to advertise its timeline and what samples it needs to our simulation
// engine so it knows what to simulate.
template <typename T> 
class Instrument {
public:
  virtual const std::vector<double> &timeline() const = 0;
  virtual const std::vector<SampleDef<T>> &samples_needed() const = 0;
  virtual const size_t number_of_payoffs() const = 0;

  virtual void payoffs(const Scenario<T> &path,
                       std::vector<T> &payoffs) const = 0;

  virtual std::unique_ptr<Instrument<T>> clone() const = 0;
  virtual ~Instrument(){}
};

// ABC interface for financial models. For us this will mostly be the
// Black-Scholes model or the Dupire model.
// First, a model needs to communicate with an instrument to initialize itself

// The generate_path function is the workhorse of a financial model object, the
// financial model
// simulates whatever observations the instrument needs to determine its
// price.

// Finally the parameter functions give clients access to the model parameters
// i.e. in Black-Scholes model this is spot/vol/div client code needs to modify
// these parameters in order to calculate the greeks
template <typename T> 
class FinancialModel {
public:
  virtual size_t simulation_dimension() const = 0;

  virtual void generate_path(const std::vector<double> &gaussian_vector,
                             Scenario<T> &path) const = 0;

  virtual void attune(const Instrument<T>& instrument) = 0;

  virtual std::unique_ptr<FinancialModel<T>> clone() const = 0;
  
  virtual ~FinancialModel(){}

  virtual const std::vector<T *> &parameters() = 0;

  size_t number_of_parameters() const {
    return const_cast<FinancialModel<T> *>(this)->parameters().size();
  }



};

// ABC interface for random number generators. Would like to benchmark many
// different RNG's to see how they perform the finance world seems to like the
// L'Ecuyer but after implementing it in Rust and comparing it to the rust RNG
// which uses "chacha", it seems like chacha absolutely blows L'Ecuyer out of
// the water
class RNG {
public:
  virtual void initialize(const size_t simulation_dimension) = 0;
  virtual void get_gaussians(std::vector<double> &gaussian_vector) = 0;

  // In order to be compatible with Sobol and some other deterministic PRNG's, we require our RNG interface to have a jump ahead 
  // this has the added benefit that a parallel simulation and non parallel simulation with the same seed will have the same result.
  // In order to avoid directly implementing this for the Mersenne Twist RNG (which is possible but tedious), 
  // the function has a default implementation that just runs the RNG and throws away the results.
  virtual void jump_ahead(const unsigned steps) = 0;

  virtual std::unique_ptr<RNG> clone() const = 0;
  virtual ~RNG(){}

  virtual size_t simulation_dimension() const = 0;
};

// Finally we have the monte carlo algorithm, which is fully generic on the
// instrument/model/rng.
inline std::vector<std::vector<double>>
monte_carlo_simulation(Instrument<double> &instrument,
                       FinancialModel<double> &model, 
                       const RNG &rng,
                       size_t num_paths) {
  
  const auto payoff_size = instrument.number_of_payoffs();
  
  std::vector<std::vector<double>> 
  results(num_paths,std::vector<double>(payoff_size));


  model.attune(instrument);

  auto c_rng = rng.clone();
  c_rng->initialize(model.simulation_dimension());

  std::vector<double> gaussian_vector(model.simulation_dimension());

  Scenario<double> path;
  initialize_path(instrument.samples_needed(), path);

  for (auto i = 0; i < num_paths; ++i) {
    c_rng->get_gaussians(gaussian_vector);
    model.generate_path(gaussian_vector, path);
    instrument.payoffs(path, results[i]);
  }


  return results;
}


inline std::vector<std::vector<double>> 
parallel_monte_carlo_simulation(
  const Instrument<double>& instrument,
  FinancialModel<double>& model,
  const RNG& rng,
  const size_t number_of_iterations) 

// The parallel version of our simulation. finally all of our hardwork will (hopefully) shine!
{
  // First some prep work. Declare a batch size, which will be the number of paths generated in each iteration.
  // We copy the RNG, each thread needs ownership of one RNG so we will need to copy it for at least all but the main thread, so why
  // not just copy for the main thread too and end the simulation with the same RNG ready to go.
  // The we allocate the result matrix.
  // finally we attune the model to the instrument -- this tells the model what it needs to simulate
  size_t batch_size = 256;
  auto crng = rng.clone();
  const auto number_of_payoffs = instrument.number_of_payoffs();
  std::vector<std::vector<double>> results(number_of_iterations, std::vector<double>(number_of_payoffs));
  model.attune(instrument);

  // now we start to prepare for the parallel simulation
  // first we set up the thread pool
  ThreadPool* pool = ThreadPool::getInstance();
  const size_t thread_count = pool->numThreads();

  // allocate a gaussian vector and a path for each thread to use
  std::vector<std::vector<double>> gaussian_matrix(thread_count + 1);
  std::vector<Scenario<double>> path_matrix(thread_count + 1);
  for(auto& vec : gaussian_matrix) vec.resize(model.simulation_dimension());
  
  // initialize each threads path vector
  for(auto& path : path_matrix){
    initialize_path(instrument.samples_needed(), path);
  }

  // each thread will have its own copy of the rng
  std::vector<std::unique_ptr<RNG>> rng_vector(thread_count + 1);
  for(auto& gen : rng_vector){
    gen = rng.clone();
    gen->initialize(model.simulation_dimension());
  }


  // now the book-keeping starts and its easy to get confused

  // this is the number of tasks that need to be assigned. 
  std::vector<std::future<bool>> future_vector;
  future_vector.reserve(number_of_iterations/batch_size + 1);

  // in each iteration the producer will increment the first_path in the batch, starting with 0,
  // and decrement the number of paths left
  size_t first_path = 0;
  size_t paths_left = number_of_iterations;

  while(paths_left > 0){
    // all but the last batch will receive batch_size paths to compute
    size_t paths_in_task = std::min(paths_left, batch_size);

    // the producer bundles up a nice bit of work and kicks it to the threadpool in 
    // a lambda, the spawn_task function returns a future<bool> for that work,
    // which is pushed into our futures_vector
    future_vector.push_back(pool->spawnTask(

      [&, first_path, paths_in_task](){
          const size_t thread_num = pool->numThreads();

          std::vector<double>& gaussian_vector = gaussian_matrix[thread_num];
          
          Scenario<double>& path = path_matrix[thread_num];
          
          // keeping track of the first_path lets us jump the rng to the correct spot
          auto& generator = rng_vector[thread_num];
          generator -> jump_ahead(first_path);

          for(size_t i = 0; i < paths_in_task; ++i){
            generator->get_gaussians(gaussian_vector);
            model.generate_path(gaussian_vector, path);
            instrument.payoffs(path, results[first_path + i]);
          }
          // return for lambda expression, not entire function. 
          return true;
    }));

    paths_left -= paths_in_task;
    first_path += paths_in_task;
  }

  for(auto& future : future_vector) future.wait(); 
  return results;
}