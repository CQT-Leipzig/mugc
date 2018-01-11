// ------------------------------------------------------------------ //
//      Author: Paul Spitzner
//     Created: 2018_01_10
//
//        File: montecarlo.hpp
// Description: Metropolis, Muca, Mugc and helpers for histograms
//              works with other system (than ljg) but same syntax
// ------------------------------------------------------------------ //

#include <vector>
#include <cmath>
// #include "rng.hpp" // already included via ljg.cpp

// ------------------------------------------------------------------ //
// METROPOLIS //
// ------------------------------------------------------------------ //

void metro_step(asystem &sys, long sweeps, long mcs, double beta = 2.5) {
  double e_move = 0.0;

  for (int s = 0; s < sweeps; ++s) {
    for (int n = 0; n < mcs; ++n) {
      e_move = sys.start_update();

      if (e_move <= 0.0) {
        sys.accept_update();
      } else if (rng() < exp(-beta*e_move)) {
        sys.accept_update();
      } else {
        sys.reject_update();
      }
    }
  }
}

// ------------------------------------------------------------------ //
// Helper functions dealing with histograms  //
// ------------------------------------------------------------------ //

// determine the first and last index with non-empty histogram entry
template <typename intT>
void get_histogram_range(const std::vector<intT> &histogram, size_t &start, size_t &end) {
  start = histogram.size();
  end = 0;
  for (size_t i=0; i<histogram.size(); i++) {
    if (histogram.at(i) > 1) {
      if (i < start) {
        start = i;
      }
      break;
    }
  }
  for (size_t i=histogram.size(); i-- > 0; ) {
    if (histogram.at(i) > 1) {
      if (i > end) {
        end = i;
      }
      break;
    }
  }
}

// update muca weights according to W^{n+1}(N) = W^{n}(N) / H^{n}(N)
template <typename intT>
void update_weights(std::vector<double> &log_w, const std::vector<intT> &histogram) {
  size_t hist_start, hist_end;
  get_histogram_range(histogram, hist_start, hist_end);
  for (size_t i=hist_start; i<=hist_end; ++i) {
    if (histogram.at(i) > 0) {
      log_w.at(i) = log_w.at(i) - log((double)(histogram.at(i)));
    }
  }
}

// error weighted update recursion.
template <typename intT>
void update_weights_advanced(std::vector<double> &fluct, std::vector<double> &ratio, std::vector<double> &log_w, const std::vector<intT> &histogram) {

  size_t hist_start, hist_end;
  get_histogram_range(histogram, hist_start, hist_end);
  if ((hist_end-hist_start)<=1) {
    update_weights(log_w, histogram);
    for (int i = 0; i < log_w.size()-1; i++) ratio.at(i) = exp(log_w.at(i+1)-log_w.at(i));
    return;
  }

  double old_fluct, this_fluct;
  for (size_t i=0; i<histogram.size()-1; i++) {
    old_fluct = fluct.at(i);
    this_fluct = histogram.at(i) + histogram.at(i+1) == 0 ? 0.0 : histogram.at(i)*histogram.at(i+1)/double(histogram.at(i)+histogram.at(i+1));
    fluct.at(i) = this_fluct+old_fluct;
    this_fluct = this_fluct+old_fluct == 0.0 ? 0.0 : this_fluct/(this_fluct+old_fluct);
    double arg = histogram.at(i+1) == 0 ? 0.0 : histogram.at(i)/double(histogram.at(i+1));
    ratio.at(i) = ratio.at(i)*pow(arg,this_fluct);
  }

  //update weights from ratio
  log_w.at(0) = 1.0;
  for (size_t i = 0; i < log_w.size()-1; i++) {
    log_w.at(i+1) = log_w.at(i) + log(ratio.at(i));
  }
}

template <typename intT>
bool check_flatness_naive(const std::vector<intT> &histogram, double &flatness, size_t n_min, size_t n_max, double crit = 0.2) {
  double min = 1e9;
  double max = 0.0;
  double mean = 0.0;
  double count = 0.0;
  for (size_t i = n_min; i<n_max; i++) {
    double tmp = histogram.at(i);
    mean += tmp;
    if (tmp < min) min = tmp;
    if (tmp > max) max = tmp;
    count += 1.0;
  }
  mean /= count;

  double ret = std::max(std::fabs(min-mean),std::fabs(max-mean));
  flatness = 100.0*ret/mean;
  if (min > (1.0-crit)*mean && max < (1.0+crit)*mean) return true;
  return false;
}

// ------------------------------------------------------------------ //
// MUCA (canonical ensemble - flat histogram in energy) //
// ------------------------------------------------------------------ //

// needed to access (integer) histogram bins via (double) energy
struct mapping {
  double min;
  double max;
  double delta;
};

template <typename intT>
inline intT& at_e(std::vector<intT> &histogram, mapping &e_map, double e) {
  if (e > e_map.max) {
    return histogram.back();
  } else if (e < e_map.min) {
    return histogram.front();
  } else {
    int bin = int((e - e_map.min)/e_map.delta);
    return histogram.at(bin);
  }
}

// log_w and hits assumed to be vectors matching the size/energy interval specified in e_map.
// number bins = (e_map.max-e_map.min)/e_map.delta
template <typename intT>
void muca_step(asystem &sys,
               size_t sweeps,
               size_t mcs,
               std::vector<double> &log_w,
               std::vector<intT> &hits,
               mapping &e_map,
               long long &tunnel_events,
               int &tunnel_pos)
{
  double e_move = 0.0;
  int current_move;

  double shift_rate = 0.5;
  double jump_rate  = 1.0;

  double e_sys = sys.system_energy();

  for (size_t s = 0; s<sweeps; s++) {
    for (size_t i = 0; i<mcs; i++) {

      double r = rng();
      if (r < shift_rate) current_move = 2;
      else                current_move = 3;

      e_move = sys.start_update(current_move);

      // reject if e_sys would go below e_min or above e_max
      if (e_sys+e_move < e_map.max && e_sys+e_move > e_map.min) {
        double p_acc = exp(at_e(log_w, e_map, e_sys+e_move) - at_e(log_w, e_map, e_sys));
        if (rng() < p_acc) {
          e_sys += e_move;
          sys.accept_update();
        } else {
          sys.reject_update();
        }
      } else {
        sys.reject_update();
      }

      at_e(hits, e_map, e_sys) += 1;

      // tunnel events, counted within 5% of energy boundaries
      if (e_sys <= e_map.min+(e_map.max-e_map.min)/20.0) {
        if (tunnel_pos == 1) tunnel_events +=1;
        tunnel_pos = 2;
      } else if (e_sys >= e_map.max-(e_map.max-e_map.min)/20.0) {
        if (tunnel_pos == 2) tunnel_events +=1;
        tunnel_pos = 1;
      }
    }
  }
}

// ------------------------------------------------------------------ //
// MUGC (grand canonical ensemble - flat histogram in density) //
// ------------------------------------------------------------------ //

// log_w and hits assumed to be vectors starting at N=0.
// set interval via n_min and n_max.
template <typename intT>
void mugc_step(asystem &sys,
               size_t sweeps,
               size_t mcs,
               double beta,
               std::vector<double> &log_w,
               std::vector<intT> &hits,
               size_t n_min,
               size_t n_max,
               long long &tunnel_events,
               int &tunnel_pos)
{
  double e_move = 0.0;
  int current_move;

  double ins_rate   = 0.25;
  double del_rate   = 0.5;
  double shift_rate = 1.0;
  // double jump_rate  = 0.0;

  double log_V = log(sys.system_volume());
  double log_N[n_max+1];
  log_N[0] = -1*std::numeric_limits<double>::max(); // log zero
  for (int i = 1; i < n_max+1; i++) log_N[i] = log(double(i));

  for (size_t s = 0; s<sweeps; s++) {
    for (size_t i = 0; i<mcs; i++) {

      size_t N = sys.particles.size();

      double r = rng();
      if (r < ins_rate)        current_move = 0;
      else if (r < del_rate)   current_move = 1;
      else if (r < shift_rate) current_move = 2;
      else                     current_move = 3;

      // cancel if N would go below n_min
      if (int(N-1)<n_min && current_move==1) {
        hits.at(N) += 1;
        continue;
      }
      // allow n_min = 0 but no other moves than insertion are possible
      if (N==0 && current_move!=0) {
        hits.at(N) += 1;
        continue;
      }
      // disable insertion for N > n_max
      if (N+1>n_max && current_move==0) {
        hits.at(N) += 1;
        continue;
      }

      e_move = sys.start_update(current_move);

      // shift and jump
      if (current_move==2 || current_move==3) {
        if (e_move <= 0.0) {
          sys.accept_update();
        } else if (rng()<exp(-beta*e_move)) {
          sys.accept_update();
        } else {
          sys.reject_update();
        }
        // insertion
      } else if (current_move==0) {
        if (rng() < exp(log_V - log_N[N+1] -beta*e_move+log_w.at(N+1)-log_w.at(N)) ) {
          sys.accept_update();
          N += 1;
        } else {
          sys.reject_update();
        }
        // deletion
      } else if (current_move==1) {
        if (rng() < exp(log_N[N]-log_V -beta*e_move+log_w.at(N-1)-log_w.at(N)) ) {
          sys.accept_update();
          N -= 1;
        } else {
          sys.reject_update();
        }
      }

      hits.at(N) += 1;

      // tunnel events
      if (N <= n_min+1) {
        if (tunnel_pos == 1) tunnel_events +=1;
        tunnel_pos = 2;
      } else if (N >= (n_max-1)*0.95) {
        if (tunnel_pos == 2) tunnel_events +=1;
        tunnel_pos = 1;
      }
    }
  }
}
