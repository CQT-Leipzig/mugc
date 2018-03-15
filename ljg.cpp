#define THREEDIM

#include <fstream>      // std::ofstream
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <sstream>      // std::stringstream
#include <list>
#include <vector>
#include <cmath>
#include <assert.h>
#include "rng.hpp"

class particle {
  public:
    double L;
    double x, y, xb, yb;
    #ifdef THREEDIM
    double z, zb;
    #endif
    int cl_i; // cluster index

    particle(double Ln, double xn, double yn, double zn = 0.0) {
      L = Ln;
      set_pos(xn, yn, zn);
    }

    inline void set_pos(double xn, double yn, double zn = 0.0) {
      x = xn-std::floor(xn/L)*L;
      y = yn-std::floor(yn/L)*L;
      #ifdef THREEDIM
      z = zn-std::floor(zn/L)*L;
      #endif
    }

    void backup_position() {
      xb = x;
      yb = y;
      #ifdef THREEDIM
      zb = z;
      #endif
    }
    void restore_position() {
      x = xb;
      y = yb;
      #ifdef THREEDIM
      z = zb;
      #endif
    }

    double get_dist_squ(particle *p) {
      double xr = std::max(x,p->x);
      double xl = std::min(x,p->x);
      double yr = std::max(y,p->y);
      double yl = std::min(y,p->y);

      double x2 = std::min((xr-xl)*(xr-xl), (L-xr+xl)*(L-xr+xl));
      double y2 = std::min((yr-yl)*(yr-yl), (L-yr+yl)*(L-yr+yl));
      double z2 = 0.0;

      #ifdef THREEDIM
      double zr = std::max(z,p->z);
      double zl = std::min(z,p->z);
      z2 = std::min((zr-zl)*(zr-zl), (L-zr+zl)*(L-zr+zl));
      #endif

      return (x2+y2+z2);
    }
};

class asystem {
  public:

    std::vector< particle* > particles;
    std::vector< std::list<particle*> > boxes;
    std::vector< std::vector<int> > neighbour_boxes;

    double L;
    double e_sys;
    double sigma, sig6, sig12, rc, rc2, rc_offset;
    double shift_range;
    // domain decomposition, number boxes nd and box length db
    int nb;
    double db;

    //update moves
    int current_index;
    particle *current_particle;
    int current_move;
    double e_move;
    int new_box, old_box;
    long long move_accepted[4];
    long long move_rejected[4];
    long long move_total[4];

    asystem(double Ln) {
      L = Ln;
      current_move = -1;
      current_particle = NULL;

      sigma = 1.0;
      double epsilon = 1.0;
      shift_range = 1.0*sigma;
      rc = 2.5*sigma;
      rc2 = rc*rc;
      sig6  = 4.0*epsilon*pow(sigma, 6.0);
      sig12 = 4.0*epsilon*pow(sigma,12.0);
      double ir6 = pow(rc2,-3.0);
      rc_offset = fabs(sig12*(ir6*ir6) - sig6*(ir6));

      // divide total volume into an integer number of sub domains
      double rcMin = 1.0*rc;
      double rcMax = 1.7*rc;
      int dMin = std::floor(L/rcMin);
      int dMax = std::ceil(L/rcMax);

      if (dMax<=dMin) {
        db = L/double(dMin);
        nb = dMin;
      } else {
        db = L;
        nb = 1;
      }

      if (nb < 4) {
        nb = 1;
        boxes.push_back(std::list<particle*> ());
        neighbour_boxes.push_back(std::vector<int> (1,0));
      } else {
        #ifdef THREEDIM
        neighbour_boxes.resize(nb*nb*nb);
        boxes.resize(nb*nb*nb);
        for (int x = 0; x < nb; x++) {
          int xr = ((x%nb+1==nb) ? x+1-nb : x+1 );
          int xl = ((x%nb-1<0)   ? x-1+nb : x-1 );
          for (int y = 0; y < nb; y++) {
            int yr = ((y%nb+1==nb) ? y+1-nb : y+1 );
            int yl = ((y%nb-1<0)   ? y-1+nb : y-1 );
            for (int z = 0; z < nb; z++) {
              int zr = ((z%nb+1==nb) ? z+1-nb : z+1 );
              int zl = ((z%nb-1<0)   ? z-1+nb : z-1 );

              int i1  = x  +  y*nb + z*nb*nb;
              int i2  = xl +  y*nb + z*nb*nb;
              int i3  = xr +  y*nb + z*nb*nb;
              int i4  = x  + yl*nb + z*nb*nb;
              int i5  = xl + yl*nb + z*nb*nb;
              int i6  = xr + yl*nb + z*nb*nb;
              int i7  = x  + yr*nb + z*nb*nb;
              int i8  = xl + yr*nb + z*nb*nb;
              int i9  = xr + yr*nb + z*nb*nb;
              int i10 = x  +  y*nb + zl*nb*nb;
              int i11 = xl +  y*nb + zl*nb*nb;
              int i12 = xr +  y*nb + zl*nb*nb;
              int i13 = x  + yl*nb + zl*nb*nb;
              int i14 = xl + yl*nb + zl*nb*nb;
              int i15 = xr + yl*nb + zl*nb*nb;
              int i16 = x  + yr*nb + zl*nb*nb;
              int i17 = xl + yr*nb + zl*nb*nb;
              int i18 = xr + yr*nb + zl*nb*nb;
              int i19 = x  +  y*nb + zr*nb*nb;
              int i20 = xl +  y*nb + zr*nb*nb;
              int i21 = xr +  y*nb + zr*nb*nb;
              int i22 = x  + yl*nb + zr*nb*nb;
              int i23 = xl + yl*nb + zr*nb*nb;
              int i24 = xr + yl*nb + zr*nb*nb;
              int i25 = x  + yr*nb + zr*nb*nb;
              int i26 = xl + yr*nb + zr*nb*nb;
              int i27 = xr + yr*nb + zr*nb*nb;

              std::vector<int> boxes_to_check = {i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27};
              neighbour_boxes[i1] = boxes_to_check;
              boxes[i1] = std::list<particle*> ();
            }
          }
        }
        #else
        for (int i = 0; i<nb*nb; i++) {
          int ir = ((i%nb+1==nb)  ?     i+1-nb : i+1 );
          int il = ((i%nb-1<0)    ?     i-1+nb : i-1 );
          int iu = ((i-nb<0)      ? nb*nb+i-nb : i-nb);
          int id = ((i+nb>=nb*nb) ? i+nb-nb*nb : i+nb);
          int iur = ((iu%nb+1==nb) ? iu+1-nb : iu+1 );
          int iul = ((iu%nb-1<0)   ? iu-1+nb : iu-1 );
          int idr = ((id%nb+1==nb) ? id+1-nb : id+1 );
          int idl = ((id%nb-1<0)   ? id-1+nb : id-1 );
          std::vector<int> boxes_to_check = {i, ir, il, iu, id, iur, iul, idr, idl};
          neighbour_boxes.push_back(boxes_to_check);
          boxes.push_back(std::list<particle*> ());
        }
        #endif

        for (int i = 0; i < 4; i++) {
          move_accepted[i] = 0;
          move_rejected[i] = 0;
          move_total[i] = 0;
        }

      }
    }

    inline int get_box(particle *p) {
      if (nb == 1) return 0;
      int nx, ny, nz;
      nx = std::floor(p->x/db);
      ny = std::floor(p->y/db);
      nz = 0;
      #ifdef THREEDIM
      nz = std::floor(p->z/db);
      #endif
      return nx + nb*ny + nb*nb*nz;
    }

    inline void set_box(particle *p) {
      int my_box = get_box(p);
      boxes[my_box].push_front(p);
    }

    inline void remove_from_box(int my_box, particle *p) {
      boxes[my_box].remove(p);
    }

    inline double particle_energy(particle *p) {
      if (p == NULL) return 0.0;

      double r2 = 0.0;
      double en = 0.0;
      double ei = 0.0;

      int my_box = get_box(p);
      std::vector<int> boxes_to_check = neighbour_boxes[my_box];
      for (auto const &b : boxes_to_check) {
        for (std::list<particle*>::iterator p2 = boxes[b].begin(); p2 != boxes[b].end(); ++p2) {
          if (boxes[b].size() > 0 && p != (*p2)) {
            r2 = p->get_dist_squ(*p2);
            if (r2 < rc2) {
              double ir6 = 1.0/(r2*r2*r2);
              ei = sig12*(ir6*ir6) - sig6*(ir6) + rc_offset;
              en += ei;
            }
          }
        }
      }
      return en;
    }

    inline double system_energy() {
      double en = 0.0;
      for (int i = 0; i<particles.size(); i++) {
        en += particle_energy(particles[i]);
      }
      e_sys = en*0.5;
      return e_sys;
    }

    inline double system_volume() {
      double volume = L*L;
      #ifdef THREEDIM
      volume *= L;
      #endif
      return volume;
    }

    // ----------------------------------------------------------------- //
    // insertion //
    // ----------------------------------------------------------------- //

    inline double ins_particle_at_pos(double x, double y, double z = 0.0) {
      assert(current_move == -1);

      particle *p = new particle (L,x,y,z);

      // u have to insert at random positions to make sure no correlation between memory and physical properties exist
      current_index = std::floor(rng()*particles.size());
      particles.emplace(particles.begin()+current_index, p);

      current_particle = p;
      current_move = 0;
      move_total[current_move] += 1;
      set_box(p);
      new_box = get_box(p);
      e_move = particle_energy(p);

      return e_move;
    }

    inline double ins_particle() {
      double x,y,z;
      z = 0.0;
      x = rng()*L;
      y = rng()*L;
      #ifdef THREEDIM
      z = rng()*L;
      #endif

      return ins_particle_at_pos(x, y, z);
    }

    inline void reject_ins() {
      assert(current_move == 0);

      move_rejected[current_move] +=1;
      remove_from_box(new_box, current_particle);
      delete current_particle;
      particles.erase(particles.begin()+current_index);
      current_particle = NULL;
      current_move = -1;
      e_move = 0.0;
    }

    inline void accept_ins() {
      assert(current_move == 0);

      move_accepted[current_move] +=1;
      current_particle = NULL;
      current_move = -1;
      // N = N+1;
      e_sys += e_move;
      e_move = 0.0;
    }

    // ----------------------------------------------------------------- //
    // deletion //
    // ----------------------------------------------------------------- //

    inline double del_particle() {
      assert(current_move == -1);

      current_index = std::floor(rng()*particles.size());
      particle *p = particles.at(current_index);

      current_particle = p;
      current_move = 1;
      move_total[current_move] +=1;
      old_box = get_box(p);
      e_move = -particle_energy(p);

      return e_move;
    }

    inline void reject_del() {
      assert(current_move==1);
      move_rejected[current_move] +=1;

      current_particle = NULL;
      current_move = -1;
      e_move = 0.0;
    }

    inline void accept_del() {
      assert(current_move==1);
      move_accepted[current_move] +=1;

      remove_from_box(old_box, current_particle);
      particles.erase(particles.begin()+current_index);
      delete current_particle;

      current_particle = NULL;
      current_move = -1;
      e_sys += e_move;
      e_move = 0.0;
    }

    // ----------------------------------------------------------------- //
    // shift //
    // ----------------------------------------------------------------- //

    inline double shift_particle() {
      int i = std::floor(rng()*particles.size());
      assert(i < particles.size());
      particle *p = particles.at(i);
      return shift_particle(p);
    }


    inline double shift_particle(particle *p) {
      assert(current_move==-1);
      p->backup_position();
      current_particle = p;
      current_move = 2;
      double e_old = particle_energy(p);
      old_box = get_box(p);
      double x,y,z;
      z = 0.0;
      x = p->x + (rng()-0.5)*shift_range;
      y = p->y + (rng()-0.5)*shift_range;
      #ifdef THREEDIM
      z = p->z + (rng()-0.5)*shift_range;
      #endif
      p->set_pos(x,y,z);
      new_box = get_box(p);
      if (new_box != old_box) set_box(p);
      double e_new = particle_energy(p);
      e_move = e_new-e_old;
      return e_move;
    }

    inline void reject_shift() {
      assert(current_move==2);
      move_rejected[current_move] +=1;
      if (new_box != old_box) remove_from_box(new_box, current_particle);
      current_particle->restore_position();
      current_particle = NULL;
      current_move=-1;
      e_move = 0.0;
    }

    inline void accept_shift() {
      assert(current_move==2);
      move_accepted[current_move] +=1;
      if (new_box != old_box) remove_from_box(old_box, current_particle);

      current_particle = NULL;
      current_move = -1;
      e_sys += e_move;
      e_move = 0.0;
    }

    // ----------------------------------------------------------------- //
    // jump //
    // ----------------------------------------------------------------- //

    inline double jump_particle() {
      int i = std::floor(rng()*particles.size());
      assert(i < particles.size());
      particle *p = particles.at(i);
      return jump_particle(p);
    }

    inline double jump_particle(particle *p) {
      assert(current_move==-1);
      p->backup_position();
      current_particle = p;
      current_move = 3;
      double e_old = particle_energy(p);
      old_box = get_box(p);
      double x, y, z;
      z = 0.0;
      x = p->x + rng()*L;
      y = p->y + rng()*L;
      #ifdef THREEDIM
      z = p->z + rng()*L;
      #endif
      p->set_pos(x,y,z);
      new_box = get_box(p);
      if (new_box != old_box) set_box(p);
      double e_new = particle_energy(p);
      e_move  = e_new-e_old;
      return e_move;
    }

    inline void reject_jump() {
      assert(current_move==3);
      move_rejected[current_move] +=1;
      if (new_box != old_box) remove_from_box(new_box, current_particle);
      current_particle->restore_position();
      current_particle = NULL;
      current_move = -1;
      e_move = 0.0;
    }

    inline void accept_jump() {
      assert(current_move==3);
      move_accepted[current_move] +=1;
      if (new_box != old_box) remove_from_box(old_box, current_particle);
      current_particle = NULL;
      current_move = -1;

      e_sys += e_move;
      e_move = 0.0;
    }

    // ----------------------------------------------------------------- //
    // update overload //
    // ----------------------------------------------------------------- //

    double start_update(int type=2) {
      if (type == 0) return ins_particle();
      else if (type == 1) return del_particle();
      else if (type == 2) return shift_particle();
      else if (type == 3) return jump_particle();
      else {
        printf("error starting update move\n");
        assert(false);
        return 0.0;
      }
    }

    void accept_update() {
      assert(current_move!=-1);
      if (current_move==0) accept_ins();
      else if (current_move==1) accept_del();
      else if (current_move==2) accept_shift();
      else if (current_move==3) accept_jump();
      else printf("error accepting update move\n");
    }

    void reject_update() {
      assert(current_move!=-1);
      if (current_move==0) reject_ins();
      else if (current_move==1) reject_del();
      else if (current_move==2) reject_shift();
      else if (current_move==3) reject_jump();
      else printf("error rejecting update move\n");
    }

    // ----------------------------------------------------------------- //
    // statistics and IO //
    // ----------------------------------------------------------------- //

    inline size_t cluster_size(double dist = 0.0) {
      if (particles.size() == 0) return 0;

      if (dist == 0.0) dist = 2.0*sigma;
      double dist2 = pow(dist,2.0);

      size_t cl = 1;
      size_t cl_index_max = 0;
      size_t cl_size_max = 1;
      size_t cl_size_temp, cl_old;
      for (size_t i = 0; i<particles.size(); i++) {
        particles.at(i)->cl_i = cl;
        for (size_t j = 0; j<i; j++) {
          cl_size_temp = 0;
          if (particles.at(i)->get_dist_squ(particles.at(j)) <= dist2) {
            cl_old = particles.at(j)->cl_i;
            size_t k = 0;
            while (k <= i) {
              if (particles.at(k)->cl_i == cl_old) {
                particles.at(k)->cl_i = cl;
                cl_size_temp++;
              }
              k++;
            }
          }
          if (cl_size_temp >= cl_size_max) {
            cl_size_max = cl_size_temp;
            cl_index_max = cl;
          }
        }
        cl++;
      }
      return cl_size_max;
    }

    std::vector<double> get_move_ar() {
      std::vector<double> v_ret;
      for (int i = 0; i < 4; i++) v_ret.push_back(double(move_accepted[i])/double(move_total[i]));
      return v_ret;
    }

    void reset_move_ar() {
      for (int i = 0; i < 4; i++) {
        move_total[i] = 0;
        move_accepted[i] = 0;
        move_rejected[i] = 0;
      }
    }

    void export_config(std::string file_name, std::string s_note = "") {

      std::ofstream file (file_name, std::ofstream::out);
      if (s_note.length() != 0) file << "#" << s_note << std::endl;
      file << "#L=" <<std::setprecision(0)<<std::fixed << L <<std::endl;
      file << "#N=" << particles.size() <<std::endl;
      file << "#x y (z)" << std::endl;
      file<<std::setprecision(20)<<std::fixed;
      for (long i = 0; i<particles.size(); i++) {
        file << particles.at(i)->x << "\t" << particles.at(i)->y;
        #ifdef THREEDIM
        file << "\t" << particles.at(i)->z;
        #endif
        file << std::endl;
      }
      file.close();
      file.clear();
    }
};
