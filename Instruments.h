#pragma once
#include "MCLib.h"

template <typename T>
class EuropeanCall : public Instrument<T>{
    double strike_;
    double expiration_;

    const size_t num_payoffs_;
    std::vector<double> my_timeline_;
    std::vector<SampleDef<T>> samples_;
    

public:
    EuropeanCall(double strike, double expiration): strike_(strike), expiration_(expiration), num_payoffs_(1) {
        my_timeline_.push_back(expiration);
        samples_.resize(1);
        samples_[0].numeraire = true;
        samples_[0].forward_maturities.push_back(expiration);
        samples_[0].discount_maturities.push_back(expiration);
    }

    std::unique_ptr<Instrument<T>> clone() const override {
        return std::make_unique<EuropeanCall<T>>(*this);
    }


    const std::vector<double>& timeline() const override {
        return my_timeline_;
    }
    
    const std::vector<SampleDef<T>>& samples_needed() const override {
        return samples_;
    }

    const size_t number_of_payoffs() const override {
        return num_payoffs_;
    }

    void payoffs(const Scenario<T> &path, std::vector<T> &payoffs) const override {
        payoffs[0] = std::max(path[0].forwards[0] - strike_, 0.0) * path[0].discounts[0] / path[0].numeraire;
    }
};

template <typename T>
class UpAndOutCall : public Instrument<T>{
    double strike_;
    double barrier_;
    double expiration_;

    double smoothing_factor_;
    std::vector<double> timeline_;
    std::vector<double> samples_needed_;
};