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


// printf into string
template<typename ... Args>
std::string string_printf(const std::string& format, Args ... args) {
  size_t size = 1 + std::snprintf(nullptr, 0, format.c_str(), args ...);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args ...);
  return std::string(buf.get(), buf.get() + size - 1);
}

template <typename intT>
void write_histogram(const std::vector<intT> &histogram, std::string file_path) {
  std::ofstream file (file_path, std::ofstream::out);
  file << std::fixed;
  file << std::scientific;
  file << "#N\tO" << std::endl;
  for (size_t i = 0; i<histogram.size(); i++) {
    file << std::setprecision(1) << i << "\t";
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
  std::string s_export_path = "";
  std::string s_weight_path = "";
  double beta = 2.5;
  double L = 10.0;
  size_t n_min = 0;
  size_t n_max = -1;
  size_t therm = 50;
  size_t ptherm = 1000;
  size_t psweeps = 1e6;
  size_t mcs = -1;
  double hist_flat_crit = 0.35;

  for (int i=0; i<argc; i++) {
    if (i != argc) {
      // check arguments without parameter first
      if (std::string(argv[i]) == "--weightsonly")    do_production = false;
      else if (std::string(argv[i]) == "--prodonly")  do_weights = false;
      else if (std::string(argv[i]) == "--snapshots")  do_snapshots = true;
      else if (i+1 != argc) {
        // arguments with a following parameter value
        if (std::string(argv[i]) == "-o")             s_export_path = argv[i+1];
        else if (std::string(argv[i]) == "-w")        s_weight_path = argv[i+1];
        else if (std::string(argv[i]) == "-b")        beta = (double)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-l")        L = (double)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-n_max")    n_max = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-n_min")    n_min = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-mcs")      mcs = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-th")       therm = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-pth")      ptherm = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-psw")      psweeps = (size_t)atof(argv[i+1]);
        else if (std::string(argv[i]) == "-f")        hist_flat_crit = (double)atof(argv[i+1]);
      }
    }
  }

  if (s_export_path.length()==0) s_export_path = "./output/";
  s_export_path += string_printf("/l%03.0f/", L);
  std::string s_dir_cmd = "mkdir -p " + s_export_path;
  if (my_rank==0) system(s_dir_cmd.c_str());
  if (my_rank==0) printf("target directory: %s\n", s_export_path.c_str());

  // ----------------------------------------------------------------- //
  // initialisation //
  // ----------------------------------------------------------------- //

  init_rng(100+my_rank);
  asystem sys = asystem(L);

  if (n_max == -1) n_max = size_t(0.8*sys.system_volume());
  if (mcs == -1) mcs=(size_t(pow(L,2.5)));
  if (my_rank==0) printf("L = %03.1f\n", L);
  if (my_rank==0) printf("beta = %f\n", beta);
  if (my_rank==0) printf("sweep size (mcs) = %ld\n", mcs);
  if (my_rank==0) printf("N range %ld to %ld\n", n_min, n_max);

  for (int i = 0; i < n_max; i++) {
    sys.ins_particle();
    sys.accept_ins();
  }
  sys.reset_move_ar();

  std::vector<double> v_log_w(n_max+3, 0.0);
  std::vector<double> v_ratio(n_max+3, 1.0);
  std::vector<double> v_fluct(n_max+3, 0.0);
  //if using advanced update scheme, initial ratios have to be obtained from initial/imported weights
  std::vector<long long> v_global_hist(n_max+3, 0);
  std::vector<long long> v_local_hist(n_max+3, 0);
  long long local_tunnel_events = 0;
  long long global_tunnel_total, global_tunnel_max, global_tunnel_min;
  int tunnel_pos = 0;
  size_t reached_bins = 1;
  size_t sweeps = 1e3;
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
      if (reached_bins < n_max - n_min) {
        sweeps = fmax(sweeps,1e5*reached_bins/mcs/num_threads);
      } else {
        sweeps *= 1.1;
      }
      if (my_rank==0) printf("%04ld, mcs: %lu, threads: %u, sweeps per thread therm: %lu, hist: %lu\n", m, mcs, num_threads, therm, sweeps);
      // thermalise
      mugc_step(sys, therm, mcs, beta, v_log_w, v_local_hist, n_min, n_max, local_tunnel_events, tunnel_pos);
      for (int i = 0; i < v_local_hist.size(); i++) v_local_hist.at(i) = 0;
      // fill histograms for iteration step
      mugc_step(sys, sweeps, mcs, beta, v_log_w, v_local_hist, n_min, n_max, local_tunnel_events, tunnel_pos);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(v_local_hist.data(), v_global_hist.data(), v_local_hist.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_tunnel_events, &global_tunnel_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

      int converged = 0;

      if (my_rank==0) {
        double flatness = 0.0;
        if (check_flatness_naive(v_global_hist, flatness, n_min, n_max, hist_flat_crit)) {
          converged = 1;
        };
        printf("\tflatness between %ld and %ld\t%d with %2.1f%%\n", n_min, n_max, converged, flatness);
        size_t start,end;
        get_histogram_range(v_global_hist, start, end);
        size_t reached_bins_new = end-start;
        if (reached_bins_new > reached_bins) reached_bins = reached_bins_new;
        printf("\ttunnel events, total: %lld, min: %lld, max: %lld\n", global_tunnel_total, global_tunnel_min, global_tunnel_max);
        printf("\treached bins %ld to %ld (now: %ld best: %ld)\n", start, end, reached_bins_new, reached_bins);
      }
      MPI_Barrier(MPI_COMM_WORLD);

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
      write_histogram(v_log_w, s_export_path + "log_w_final.dat");
      write_histogram(v_global_hist, s_export_path + "hits_weight_iteration.dat");
    }
  }

  // ----------------------------------------------------------------- //
  // import weights if no iteration wanted and weights already present //
  // ----------------------------------------------------------------- //

  if (!do_weights) {
    if (s_weight_path.length()==0) s_weight_path = s_export_path + "/log_w_final.dat";
    if (my_rank==0) {
      printf("importing weights from: %s\n", s_weight_path.c_str());
      std::ifstream test_file(s_weight_path);
      if (!test_file.good()) {
        printf("file broken or does not exist, aborting\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
      read_histogram(v_log_w, s_weight_path);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(v_log_w.data(), v_log_w.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // ----------------------------------------------------------------- //
  // prduction run //
  // ----------------------------------------------------------------- //

  bool exported_state[n_max+1];
  for (int i = 0; i < n_max+1; i++) exported_state[i] = !do_snapshots;

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
    timeseries << "#N\t" << "LargestCluster\t" << "Energy\n";
    for (size_t i = 0; i<ptherm; i+=num_threads) {
      mugc_step(sys, 1, mcs, beta, v_log_w, v_local_hist, n_min, n_max, local_tunnel_events, tunnel_pos);
      double n = sys.particles.size();
      double c = sys.cluster_size();
      double e = sys.system_energy();
      timeseries << std::fixed << std::setprecision(1);
      timeseries << n << "\t";
      timeseries << c << "\t";
      timeseries << std::fixed << std::setprecision(7);
      timeseries << e << "\n";
    }
    timeseries.close();
    timeseries.clear();

    if (my_rank==0) printf("starting production run\n");

    float print_hours = 1.0;
    clock_t delta_t = clock();
    int printed = 0;
    local_tunnel_events = 0;
    for (int i = 0; i < v_local_hist.size(); i++) v_local_hist.at(i) = 0;

    timeseries.open(s_export_path+string_printf("/production_ts/prod_%03d.dat",my_rank), std::ofstream::out);
    timeseries << std::fixed << std::setprecision(1);
    timeseries << "#N\t" << "LargestCluster\t" << "Energy\n";
    for (size_t i = 0; i<psweeps; i+=num_threads) {
      mugc_step(sys, 1, mcs, beta, v_log_w, v_local_hist, n_min, n_max, local_tunnel_events, tunnel_pos);
      size_t n = sys.particles.size();
      size_t c = sys.cluster_size();
      double e = sys.system_energy();
      timeseries << std::fixed << std::setprecision(1);
      timeseries << n << "\t";
      timeseries << c << "\t";
      timeseries << std::fixed << std::setprecision(7);
      timeseries << e << "\n";

      if (my_rank==0 && !exported_state[n]) {
        sys.export_config(s_export_path+string_printf("/snapshots/n_%05d.dat",int(n)));
        exported_state[int(n)] = true;
      }

      clock_t now = clock();
      if ((float)(now - delta_t)/CLOCKS_PER_SEC/3600.0 >= print_hours || printed < 5) {
        printed+=1;
        delta_t = now;
        if (my_rank==0) printf("%ld / %ld ~ %3.1f%%\n", i, psweeps, i/float(psweeps)*100.0f);
        if (my_rank==0) print_time_since(start_time);
      }
    }
    timeseries.close();
    timeseries.clear();
  }

  write_histogram(v_local_hist, s_export_path + string_printf("/production_hist/hist_%03d.dat", my_rank));

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(v_local_hist.data(), v_global_hist.data(), v_local_hist.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tunnel_events, &global_tunnel_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank==0) {
    write_histogram(v_global_hist, s_export_path + "hits_production.dat");
    printf("\nproduction finished\n");
    printf("tunnel events, total: %lld, min: %lld, max: %lld\n", global_tunnel_total, global_tunnel_min, global_tunnel_max);
    print_time_since(start_time);
  }

  MPI_Finalize();

  return 0;
}
