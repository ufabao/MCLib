#pragma once
#include "MCLib.h"
#include <memory>

template <typename T> class EuropeanCall : public Instrument<T> {
  double strike_;
  double expiration_;

  const size_t num_payoffs_;
  std::vector<double> my_timeline_;
  std::vector<SampleDef<T>> samples_;

public:
  EuropeanCall(double strike, double expiration)
      : strike_(strike), expiration_(expiration), num_payoffs_(1) {
    my_timeline_.push_back(expiration);
    samples_.resize(1);
    samples_[0].numeraire = true;
    samples_[0].forward_maturities.push_back(expiration);
    samples_[0].discount_maturities.push_back(expiration);
  }

  std::unique_ptr<Instrument<T>> clone() const override {
    return std::make_unique<EuropeanCall<T>>(*this);
  }

  const std::vector<double> &timeline() const override { return my_timeline_; }

  const std::vector<SampleDef<T>> &samples_needed() const override {
    return samples_;
  }

  const size_t number_of_payoffs() const override { return num_payoffs_; }

  void payoffs(const Scenario<T> &path,
               std::vector<T> &payoffs) const override {
    payoffs[0] = std::max(path[0].forwards[0] - strike_, 0.0) *
                 path[0].discounts[0] / path[0].numeraire;
  }
};

template <typename T> class UpAndOutCall : public Instrument<T> {
  double strike_;
  double expiration_;
  double barrier_;
  double smoothing_factor_;

  static constexpr double one_hour = 0.000114469;

  vector<double> timeline_;
  vector<SampleDef<T>> samples_needed_;

public:
  UpAndOutCall(const double strike, const double expiry, const double barrier,
               const double monitor_frequency, const double smoothing_factor)
      : strike_(strike), expiration_(expiry), barrier_(barrier),
        smoothing_factor_(smoothing_factor) {
    timeline_.push_back(0.0);
    auto t = monitor_frequency;
    while (expiry - t > one_hour) {
      timeline_.push_back(monitor_frequency);
      t += monitor_frequency;
    }

    timeline_.push_back(expiry);

    const size_t n = timeline_.size();

    samples_needed_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      samples_needed_[i].numeraire = false;
      samples_needed_[i].forward_maturities.push_back(timeline_[i]);
    }

    samples_needed_.back().numeraire = true;
  }


  unique_ptr<Instrument<T>> clone() const override {
    return make_unique<UpAndOutCall<T>>(*this);
  }

    const vector<double>& timeline() const override {
        return timeline_;
    }

    const vector<SampleDef<T>>& samples_needed() const override {
        return samples_needed_;
    } 

    const size_t number_of_payoffs() const override {
        return 2;
    }

    void payoffs(const Scenario<T>& path, vector<double>& payoffs) const override {
        const double smooth = double(path[0].forwards[0] * smoothing_factor_);
        const double two_smooth = 2 * smooth;
        const double bar_smooth = barrier_ + smooth;

        T alive(1.0);
        
        for(const auto& sample : path){
            if(sample.forwards[0] > bar_smooth){
                alive = T(0.0);
                break;
            }

            if(sample.forwards[0] > barrier_ - smooth){
                alive *= (bar_smooth - sample.forwards[0]) / two_smooth;
            }
        }

        payoffs[1] = max(path.back().forwards[0] - strike_, 0.0) / path.back().numeraire;
        payoffs[0] = alive * payoffs[1];
    }

};