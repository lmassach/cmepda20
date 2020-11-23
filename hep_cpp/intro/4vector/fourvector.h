#ifndef FOURVECTOR_H
#define FOURVECTOR_H

class FourVector {
  public:
    FourVector() {}
    FourVector(double px, double py, double pz, double e) :
        px_(px), py_(py), pz_(pz), e_(e) {}

    double px() const { return px_; }
    double py() const { return py_; }
    double pz() const { return pz_; }
    double e() const { return e_; }

    double m() const;
    double pt() const;
    double p() const;
    double phi() const;
    double theta() const;
    double eta() const;
    double y() const;
    double gamma() const;
    double beta() const;

    FourVector operator+(const FourVector &other);

  protected:
    double px_, py_, pz_, e_;
};

#endif // FOURVECTOR_H
