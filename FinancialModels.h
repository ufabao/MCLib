#pragma once
#include "MCLib.h"
#include <cmath>

// class is defined generically over the number type so that later on we can
// implement autodiff for the greeks
template <typename T> class BlackScholesModel : public FinancialModel<T> {
  T spot_;
  T vol_;
  T rate_;
  T div_;

  //the instrument will tell us what dates and what samples it needs
  std::vector<double> timeline_;
  const std::vector<SampleDef<T>> *samples_needed_;

  std::vector<T *> parameters_;

  std::vector<T> underlying_drifts_;
  std::vector<T> underlying_stds_;

  std::vector<T> numeraires_;
  std::vector<std::vector<T>> forward_factors_;
  std::vector<std::vector<T>> discount_factors_;

public:
  template <typename U>
  BlackScholesModel(const U spot, const U vol,
                    const U rate = U{0.0}, const U div = U{0.0})
      : spot_(spot), rate_(rate), div_(div), vol_(vol), parameters_(4) 
  {
    set_parameter_pointers();
  }

  T spot() const{
    return spot_;
  }

  const T vol() const {
    return vol_;
  }

  const T rate() const {
    return rate_;
  }

  const T div() const {
    return div_;
  }

  const std::vector<T*>& parameters() override {
    return parameters_;
  }


private:
  void set_parameter_pointers(){
    parameters_[0] = &spot_;
    parameters_[1] = &vol_;
    parameters_[2] = &rate_;
    parameters_[3] = &div_;
  }

public:
  std::unique_ptr<FinancialModel<T>> clone() const override{
    auto clone = std::make_unique<BlackScholesModel<T>>(*this);
    clone -> set_parameter_pointers();
    return clone;
  }

  void allocate(const std::vector<double>& instrument_timeline, const std::vector<SampleDef<T>>& samples_needed) override {
    timeline_.clear();
    timeline_.push_back(0.0);
    for(const auto& time : instrument_timeline) if(time > 0.0) timeline_.push_back(time);

    samples_needed_ = &samples_needed;
    underlying_drifts_.resize(timeline_.size() - 1);
    underlying_stds_.resize(timeline_.size() - 1);

    const size_t n = instrument_timeline.size();
    numeraires_.resize(n);
    discount_factors_.resize(n);
    for(auto j = 0; j < n; ++j){
        discount_factors_[j].resize(samples_needed[j].discount_maturities.size());
    }
    
    forward_factors_.resize(n);
    for(auto j = 0; j < n; ++j){
        forward_factors_[j].resize(samples_needed[j].forward_maturities.size());
    }
  }

  void initialize(const std::vector<double>& instrument_timeline, const std::vector<SampleDef<T>>& samples_needed) override {
    // We want to precompute everything that does not rely on the simulation
    const T mu = rate_ - div_;

    // pre compute the drifts and devs 
    const size_t n = timeline_.size() - 1;
    for(auto i = 0; i < n; ++i){
        const double dt = timeline_[i+1] - timeline_[i];
        underlying_stds_[i] = vol_ * std::sqrt(dt);
        underlying_drifts_[i] = (mu - 0.5 * vol_ * vol_)*dt;
    }

    // pre compute the forward and discount rates
    const size_t m = instrument_timeline.size();
    for(auto i = 0; i < m; ++i){
        if(samples_needed[i].numeraire){
            numeraires_[i] = std::exp(rate_ * instrument_timeline[i]);
        }

        const size_t nFF = samples_needed[i].forward_maturities.size();
        for(auto j = 0; j < nFF; ++j){
            forward_factors_[i][j] = std::exp(mu * (samples_needed[i].forward_maturities[j] - instrument_timeline[i]));
        }

        const size_t nDF = samples_needed[i].discount_maturities.size();
        for(auto j = 0; j < nDF; ++j){
            discount_factors_[i][j] = std::exp(-rate_ * (samples_needed[i].discount_maturities[j] - instrument_timeline[i]));
        }
    }
  }

  size_t simulation_dimension() const override {
    return timeline_.size() - 1;
  }

private:
  inline void fillScen(const size_t idx, const T& spot, MarketSample<T>& sample, const SampleDef<T>& def) const {
    if(def.numeraire){
        sample.numeraire = numeraires_[idx];
    }

    std::transform(forward_factors_[idx].begin(), forward_factors_[idx].end(), sample.forwards.begin(), 
                        [&spot](const T& ff){return spot * ff;});

    std::copy(discount_factors_[idx].begin(), discount_factors_[idx].end(), sample.discounts.begin());
  }

public:
  void generate_path(const std::vector<double>& gaussian_vector, Scenario<T>& path) const override {
    T spot = spot_;


    const size_t n = timeline_.size() - 1;
    for(auto i = 0; i < n; ++i){
        spot = spot * std::exp(underlying_drifts_[i]+ underlying_stds_[i] * gaussian_vector[i]);
        fillScen(i, spot, path[i], (*samples_needed_)[i]);
    }
  }
};