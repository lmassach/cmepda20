#include <cmath>
#include "fourvector.h"

double FourVector::m() const {
    return sqrt(e_*e_ - px_*px_ - py_*py_ - pz_*pz_);
}

double FourVector::pt() const {
    return sqrt(px_*px_ + py_*py_);
}

double FourVector::p() const {
    return sqrt(px_*px_ + py_*py_ + pz_*pz_);
}

double FourVector::phi() const {
    return atan2(py_, px_);
}

double FourVector::theta() const {
    return atan2(pt(), pz_);
}

double FourVector::eta() const {
    return -log(tan(theta() / 2.0));
}

double FourVector::y() const {
    return 0.5 * log((e_ + pz_) / (e_ - pz_));
}

double FourVector::gamma() const {
    return e_ / m();
}

double FourVector::beta() const {
    return p() / e_;
}

FourVector FourVector::operator+(const FourVector &other) {
    return FourVector(px_+other.px_, py_+other.py_, pz_+other.pz_, e_+other.e_);
}
