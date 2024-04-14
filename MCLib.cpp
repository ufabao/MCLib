#include "MCLib.h"
#include "Instruments.h"
#include "FinancialModels.h"
#include "RNGs.h"


int main(){
    BlackScholesModel<double> model{100.0, 0.2};
    EuropeanCall<double> call{100.0, 1.0};
    MersenneTwistRNG rng;

    auto result = monte_carlo_simulation(call, model, rng, 100000);
    auto mean = std::accumulate(result.begin(), result.end(), 0.0l,
                [](auto sum, auto v){return sum + v[0] / 100000;});

    std::cout << mean << "\n";
}