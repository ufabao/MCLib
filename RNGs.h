#pragma once
#include "MCLib.h"
#include <random>
#include <pcg-cpp-0.98/include/pcg_random.hpp>

// A classic Mersenne twist RNG. 
class MersenneTwistRNG : public RNG {
  int seed_{42};
  std::mt19937_64 generator_;
  std::normal_distribution<double> distribution_{0.0, 1.0};
  size_t dimension_;

  // store a cache for antithetic sampling
  std::vector<double> cached_values_;
  bool antithetic_flag_{false};

public:
  MersenneTwistRNG(int seed = 42): seed_(seed), generator_(seed) {}

  // introduce the RNG to the model so we know how many gaussians our model plans on consuming each iteration
  void initialize(const size_t simulation_dimension) override {
    dimension_ = simulation_dimension;
    cached_values_.resize(dimension_);
  }
  
  // The workhorse of our RNG. Given a preallocated vector we populate it with gaussian vectors. We are using antithetic sampling
  // so if the flag is false we generate new gaussians which we cache and pass to the model, if the flag is true then 
  // we take our cached gaussians and give the model their negation
  void get_gaussians(std::vector<double> &gaussian_vector) override {
    if (antithetic_flag_) {
      std::transform(cached_values_.begin(), cached_values_.end(),
                     gaussian_vector.begin(),
                     [](const double n) { return -n; });
      antithetic_flag_ = false;
    } else {
      std::generate(cached_values_.begin(), cached_values_.end(),
                    [&] { return distribution_(generator_); });
      std::copy(cached_values_.begin(), cached_values_.end(),
                gaussian_vector.begin());
      antithetic_flag_ = true;
    }
  }

  std::unique_ptr<RNG> clone() const override {
    return std::make_unique<MersenneTwistRNG>(*this);
  }

  virtual void jump_ahead(const unsigned steps) override {
    std::vector<double> trash(simulation_dimension());
    for(auto i = 0; i < steps; ++i) get_gaussians(trash);
  }


  size_t simulation_dimension() const override { return dimension_; }
};


// Apparently the PCG family of RNG's are the state of the art for monte carlo simulations, although it doesn't seem like many finance
// books/repositories use them. 
class PCGRNG : public RNG{
  long unsigned seed_{42};
  pcg32 generator_;
  std::normal_distribution<double> distribution_{0.0, 1.0};

  size_t dimension_;
  std::vector<double> cached_values_;
  bool antithetic_flag_{false};

public:
  PCGRNG(int seed = 42): seed_(seed), generator_{seed_} {}

  void initialize(const size_t simulation_dimension) override {
    dimension_ = simulation_dimension;
    cached_values_.resize(dimension_);
  }

   void get_gaussians(std::vector<double> &gaussian_vector) override {
    if (antithetic_flag_) {
      std::transform(cached_values_.begin(), cached_values_.end(),
                     gaussian_vector.begin(),
                     [](const double n) { return -n; });
      antithetic_flag_ = false;
    } else {
      std::generate(cached_values_.begin(), cached_values_.end(),
                    [&] { return distribution_(generator_); });
      std::copy(cached_values_.begin(), cached_values_.end(),
                gaussian_vector.begin());
      antithetic_flag_ = true;
    }
  }

  void jump_ahead(const unsigned steps) override {
    generator_.advance((steps * simulation_dimension())/2);
  }

  std::unique_ptr<RNG> clone() const override {
    return std::make_unique<PCGRNG>(*this);
  }


  size_t simulation_dimension() const override { return dimension_; }

};