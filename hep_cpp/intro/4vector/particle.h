#ifndef PARTICLE_H
#define PARTICLE_H

#include "fourvector.h"

class Particle : public FourVector {
  public:
    // Particle() {} // Non necessario, esiste di default
    Particle(double charge, FourVector p) : FourVector(p), charge_(charge) {}

    double charge() const { return charge_; }

  protected:
    double charge_;
};

#endif // PARTICLE_H
