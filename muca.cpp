#include "ljg.cpp"
// #include "rng.hpp" // already included via ljg.cpp
#include "montecarlo.hpp"

#include <mpi.h>
#include <string>
#include <memory>       // std::unique_ptr
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <sstream>      // std::stringstream
#include <ctime>        // std::time


// printf into string, only works with at least one placeholder
template<typename ... Args>
std::string string_printf(const std::string& format, Args ... args) {
  size_t size = 1 + std::snprintf(nullptr, 0, format.c_str(), args ...);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args ...);
  return std::string(buf.get(), buf.get() + size - 1);
}

template <typename intT>
void write_histogram(const std::vector<intT> &histogram, double e_min, double e_delta, std::string file_path) {
  std::ofstream file (file_path, std::ofstream::out);
  file << std::fixed;
  file << std::scientific;
  file << "#E\tO" << std::endl;
  for (size_t i = 0; i<histogram.size(); i++) {
    file << std::setprecision(5) << e_min+i*e_delta << "\t";
    file << std::setprecision(15) << histogram.at(i) << std::endl;
  }
  file.close();
  file.clear();
}

template <typename intT>
void read_histogram(std::vector<intT> &histogram, std::string file_path) {
  std::string linse_string;
  std::ifstream input_stream (file_path);
  std::vector<double> v_entries;
  std::vector<double> v_bins;

  if (input_stream.is_open()) {
    while (std::getline(input_stream, linse_string)) {
      std::stringstream line_stream(linse_string);
      std::string ptrn = "#";
      std::size_t found_start = linse_string.find(ptrn);
      if (found_start==std::string::npos) {
        double bin, entry;
        line_stream >> bin >> entry;
        v_bins.push_back(bin);
        v_entries.push_back(entry);
      }
    }
    if (!input_stream.eof()) printf("end of file was not reached: %s\n", file_path.c_str());
    input_stream.close();
  } else {
    printf("unable to open input file: %s\n", file_path.c_str());
  }

  //histogram from file may not start at zero
  for (size_t i = 0; i < histogram.size(); i++) {
    histogram.at(i) = 0.0;
    for (size_t j = 0; j < v_bins.size(); j++) {
      if (size_t(v_bins.at(j)) == i) histogram.at(i) = intT(v_entries.at(j));
    }
  }

  // for (int i = 0; i < histogram.size(); i++) {
  //   printf("%d %f\n", i, histogram.at(i));
  // }
}

void print_time_since(const clock_t start) {
  clock_t end = clock();
  float seconds_used = (float)(end - start) / CLOCKS_PER_SEC;
  int days_used = int(seconds_used/86400.0);
  seconds_used = fmod(seconds_used,86400.0);
  int hours_used = int(seconds_used/3600.0);
  seconds_used = fmod(seconds_used,3600.0);
  int minutes_used = int(seconds_used/60.0);
  seconds_used = fmod(seconds_used,60.0);
  printf("\t%02dd %02dh %02dm %02ds\n", days_used, hours_used, minutes_used, int(seconds_used));
}

int main(int argc, char *argv[]) {

  // initialize MPI
  int my_rank = 0;
  int num_threads = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

  // ----------------------------------------------------------------- //
  // arguments //
  // ----------------------------------------------------------------- //

  bool do_production = true;
  bool do_weights = true;
  bool do_snapshots = false;
  bool do_iterationfiles = false;
  std::string s_export_path = "";
  size_t N = 100;
  double e_min = 1.0;
  double e_max = 0.0;
  double e_delta = 1.0;
  double beta_init = 0.0;
  double rho = 0.027857;
  size_t w_therm = 50;
  size_t w_sweeps = 1e3;
  size_t p_therm = 1e4;
  size_t p_sweeps = 1e6;
  size_t mcs = -1;
  size_t pmcs = -1;
  double hist_flat_crit = 0.35;

  for (int i=0; i<argc; i++) {
    if (i != argc) {
      // check arguments without parameter first
      if (std::string(argv[i]) == "--weightsonly")          do_production = false;
      else if (std::string(argv[i]) == "--snapshots")       do_snapshots = true;
      else if (std::string(argv[i]) == "--iterationfiles")  do_iterationfiles = true;
      else if (i+1 != argc) {
        // arguments with a following parameter value
        if (std::string(argv[i]) == "-o")             s_export_path = argv[i+1];
        else if (std::string(argv[i]) == "-n")        N = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-rho")      rho = (double)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-emax")     e_max = (double)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-emin")     e_min = (double)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-mcs")      mcs = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-pmcs")     pmcs = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-wth")      w_therm = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-wsw")      w_sweeps = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-pth")      p_therm = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-psw")      p_sweeps = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-f")        hist_flat_crit = (double)atof(argv[i+1]);
      }
    }
  }

  // ----------------------------------------------------------------- //
  // initialisation //
  // ----------------------------------------------------------------- //

  if (s_export_path.length()==0) s_export_path = "./output/";
  s_export_path += string_printf("/n%04d/", N);
  std::string s_dir_cmd = "mkdir -p " + s_export_path;
  if (my_rank==0) system(s_dir_cmd.c_str());
  if (my_rank==0) printf("target directory: %s\n", s_export_path.c_str());
  if (my_rank==0 && do_iterationfiles) s_dir_cmd = "mkdir -p " + s_export_path + "/weight_hist/";
  if (my_rank==0 && do_iterationfiles) system(s_dir_cmd.c_str());

  if (e_min == 1.0) e_min = -2.0*double(N);
  if (mcs == -1) mcs=(size_t(pow(N,2)));
  double L = pow(double(N)/rho,1.0/2.0);
  #ifdef THREEDIM
  L = pow(double(N)/rho,1.0/3.0);
  #endif
  if (my_rank==0) printf("N = %ld, L = %03.1f\n", N, L);
  if (my_rank==0) printf("rho = %f\n", rho);
  if (my_rank==0) printf("sweep size (mcs) = %ld\n", mcs);

  init_rng(100+my_rank);
  asystem sys = asystem(L);

  for (int i = 0; i < N; i++) {
    sys.ins_particle();
    sys.accept_ins();
  }
  // avoid unphysical systems with very close particles
  // metro_step(sys, 1e2, N*N, 0.1);
  sys.init_gas_state();
  sys.reset_move_ar();

  mapping e_map;
  e_map.max = e_max;
  e_map.min = e_min;
  e_map.delta = e_delta;
  std::vector<double> v_log_w((e_map.max - e_map.min)/e_map.delta+1, 0.0);
  std::vector<double> v_ratio(v_log_w.size(), 1.0);
  std::vector<double> v_fluct(v_log_w.size(), 0.0);
  for (int i = 0; i < v_log_w.size(); i++) v_log_w.at(i) = -beta_init*i;
  //if using advanced update scheme, initial ratios have to be obtained from initial/imported weights
  std::vector<long long> v_global_hist(v_log_w.size(), 0);
  std::vector<long long> v_local_hist(v_log_w.size(), 0);
  long long local_tunnel_events = 0;
  long long global_tunnel_total, global_tunnel_max, global_tunnel_min;
  int tunnel_pos = 0;
  size_t reached_bins = 1;
  clock_t start_time = clock();

  // ----------------------------------------------------------------- //
  // weight iteration //
  // ----------------------------------------------------------------- //

  if (do_weights) {
    if (my_rank==0) printf("weight iteration\n");
    size_t iter_max = 1e5;
    for (size_t m = 0; m < iter_max; m++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(v_log_w.data(), v_log_w.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // update amount of statistics
      if (reached_bins < v_log_w.size()-1) {
        w_sweeps = fmax(w_sweeps,1e5*reached_bins/mcs/num_threads);
      } else {
        w_sweeps *= 1.1;
      }
      if (my_rank==0) printf("%04ld, mcs: %lu, threads: %u, sweeps per thread therm: %lu, hist: %lu\n", m, mcs, num_threads, w_therm, w_sweeps);
      // thermalise
      muca_step(sys, w_therm, mcs, v_log_w, v_local_hist, e_map, local_tunnel_events, tunnel_pos);
      for (int i = 0; i < v_local_hist.size(); i++) v_local_hist.at(i) = 0;
      // fill histograms for iteration step
      muca_step(sys, w_sweeps, mcs, v_log_w, v_local_hist, e_map, local_tunnel_events, tunnel_pos);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(v_local_hist.data(), v_global_hist.data(), v_local_hist.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

      int converged = 0;

      if (my_rank==0) {
        double flatness = 0.0;
        if (check_flatness_naive(v_global_hist, flatness, 0, v_log_w.size()-1, hist_flat_crit)) {
          converged = 1;
        };
        printf("\tflatness between %d and %ld\t%d with %2.1f%%\n", 0, v_log_w.size()-1, converged, flatness);
        size_t start,end;
        get_histogram_range(v_global_hist, start, end);
        size_t reached_bins_new = end-start;
        if (reached_bins_new > reached_bins) reached_bins = reached_bins_new;
        printf("\ttunnel events, total: %lld, min: %lld, max: %lld\n", global_tunnel_total, global_tunnel_min, global_tunnel_max);
        printf("\treached bins %ld to %ld (now: %ld best: %ld)\n", start, end, reached_bins_new, reached_bins);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      if (my_rank == 0 && do_iterationfiles) {
        write_histogram(v_log_w, e_map.min, e_map.delta, s_export_path + string_printf("/weight_hist/log_w_%04d.dat",m));
        write_histogram(v_global_hist, e_map.min, e_map.delta, s_export_path + string_printf("/weight_hist/hits_%04d.dat",m));
      }

      // Broadcast convergence state to all ranks
      MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&reached_bins, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (converged == 1) break;

      if (my_rank == 0) {
        update_weights(v_log_w, v_global_hist);
        // update_weights_advanced(v_fluct, v_ratio, v_log_w, v_global_hist);
        print_time_since(start_time);
      }
    }

    if (my_rank == 0) {
      write_histogram(v_log_w, e_map.min, e_map.delta, s_export_path + "log_w_final.dat");
      write_histogram(v_global_hist, e_map.min, e_map.delta, s_export_path + "hits_weight_iteration.dat");
    }
  }

  // ----------------------------------------------------------------- //
  // prduction run //
  // ----------------------------------------------------------------- //

  std::vector<int> exported_state(v_log_w.size(), int(!do_snapshots));

  if (do_production) {
    if (my_rank==0) {
      printf("thermalising for production run\n");
      s_dir_cmd = "mkdir -p " + s_export_path + "/production_ts/";
      system(s_dir_cmd.c_str());
      s_dir_cmd = "mkdir -p " + s_export_path + "/production_hist/";
      system(s_dir_cmd.c_str());
      if (do_snapshots) s_dir_cmd = "mkdir -p " + s_export_path + "/snapshots/";
      if (do_snapshots) system(s_dir_cmd.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::ofstream timeseries;
    timeseries.open(s_export_path+string_printf("/production_ts/therm_%03d.dat",my_rank), std::ofstream::out);
    timeseries << std::fixed << std::setprecision(1);
    timeseries << "#Energy\t" << "LargestCluster\n";
    for (size_t i = 0; i<p_therm; i+=num_threads) {
      muca_step(sys, 1, mcs, v_log_w, v_local_hist, e_map, local_tunnel_events, tunnel_pos);
      double e = sys.system_energy();
      size_t c = sys.cluster_size();
      timeseries << std::fixed << std::setprecision(7);
      timeseries << e << "\t";
      timeseries << std::fixed << std::setprecision(1);
      timeseries << c << "\n";
    }
    timeseries.close();
    timeseries.clear();

    if (pmcs != -1) mcs = pmcs;
    if (my_rank==0) {
      if (pmcs != -1) printf("adjusting mcs for production run to %ld\n", mcs);
      printf("starting production run\n");
    }


    float print_hours = 6.0;
    clock_t delta_t = clock();
    int printed = 0;
    int crossed_tenth = p_sweeps/10;
    local_tunnel_events = 0;
    for (int i = 0; i < v_local_hist.size(); i++) v_local_hist.at(i) = 0;

    timeseries.open(s_export_path+string_printf("/production_ts/prod_%03d.dat",my_rank), std::ofstream::out);
    timeseries << std::fixed << std::setprecision(1);
    timeseries << "#Energy\t" << "LargestCluster\n";
    for (size_t i = 0; i<p_sweeps; i+=num_threads) {
      muca_step(sys, 1, mcs, v_log_w, v_local_hist, e_map, local_tunnel_events, tunnel_pos);
      double e = sys.system_energy();
      size_t c = sys.cluster_size();
      timeseries << std::fixed << std::setprecision(7);
      timeseries << e << "\t";
      timeseries << std::fixed << std::setprecision(1);
      timeseries << c << "\n";

      // snapshots from rank 0 and logging
      if (my_rank == 0) {
        if (!at_e(exported_state, e_map, e)) {
          sys.export_config(s_export_path+string_printf("/snapshots/e_%05.0f.dat",e));
          at_e(exported_state, e_map, e) = 1;
        }

        clock_t now = clock();
        bool print_now = false;
        if (i >= crossed_tenth) {
          crossed_tenth += p_sweeps/10;
          print_now = true;
        }
        if ((float)(now - delta_t)/CLOCKS_PER_SEC/3600.0 >= print_hours) {
          print_now = true;
        }
        if (printed < 5) {
          print_now = true;
        }
        if (print_now) {
          printed+=1;
          delta_t = now;
          printf("%ld / %ld ~ %3.1f%%\n", i, p_sweeps, i/float(p_sweeps)*100.0f);
          print_time_since(start_time);
        }
      }
    }
    timeseries.close();
    timeseries.clear();
  }

  write_histogram(v_local_hist, e_map.min, e_map.delta, s_export_path + string_printf("/production_hist/hist_%03d.dat", my_rank));

  if (my_rank == 0) printf ("waiting for all threads to finish");

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(v_local_hist.data(), v_global_hist.data(), v_local_hist.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank==0) {
    write_histogram(v_global_hist, e_map.min, e_map.delta, s_export_path + "hits_production.dat");
    printf("\nproduction finished\n");
    printf("tunnel events, total: %lld, min: %lld, max: %lld\n", global_tunnel_total, global_tunnel_min, global_tunnel_max);
    print_time_since(start_time);
  }

  MPI_Finalize();

  return 0;
}
