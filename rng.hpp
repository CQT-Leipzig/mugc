// ------------------------------------------------------------------ //
//      Author: Paul Spitzner
//     Created: 2018_01_11
//
//        File: rng.hpp
// Description: makes random numbers globally available via a
//              rng() function, initialise only once!
//              return type is equal to the one specified in
//              the distribution, here: double
//              BEWARE: casting to lower precision WILL
//              influence the open/closed interval of returned numbers
// ------------------------------------------------------------------ //

#include <functional>
#include <random>

std::mt19937 rng_engine(314);
auto rng = bind(std::uniform_real_distribution<double>(0.0,1.0), std::ref(rng_engine));

// sets the seed for rng()
// also resets the state/internal counter of the engine.
void init_rng(int rng_seed) {
  rng_engine.seed(rng_seed);
  rng_engine.discard(70000);
}
