#ifndef FOURVECTOR_H
#define FOURVECTOR_H

#include <cmath>

// Gli oggetti template non possono essere separati in .h e .cpp perché
// il codice non può essere convertito in assembly finché non si
// stabilisce cos'è T, il che avviene solo quando usiamo la classe.

template<class T> class FourVector {
  public:
    FourVector(T px, T py, T pz, T e) : px_(px), py_(py), pz_(pz), e_(e) {}

    T px() const { return px_; }
    T py() const { return py_; }
    T pz() const { return pz_; }
    T e() const { return e_; }

    T pt() const { return sqrt(px_*px_ + py_*py_); }
    T m() const { return sqrt(e_*e_ - px_*px_ - py_*py_ - pz_*pz_); }
    T phi() const { return atan2(py_, px_); }
    T theta() const { return atan2(pt(), pz_); }
    T eta() const { return -log(tan(theta() / 2)); }

    // template <class T2> FourVector<T> operator+(const FourVector<T2> &other) const {
    //     // Dobbiamo usare gli accessors su other perché FourVector<T2> non è
    //     // lo stesso tipo di FourVector<T>
    //     return FourVector<T>(px_ + other.px(), py_ + other.py(),
    //                          pz_ + other.pz(), e_ + other.e());
    // }

    // Casting
    template <class T2> operator FourVector<T2>() const {
        return FourVector<T2>(px_, py_, pz_, e_);
    }

    FourVector<T> operator+(const FourVector<T> &other) const {
        return FourVector<T>(px_ + other.px_, py_ + other.py_, pz_ + other.pz_,
                             e_ + other.e_);
    }
  protected:
    T px_, py_, pz_, e_;
};

template<class T> class FourVectorPtEtaPhiM {
  public:
    FourVector(T pt, T eta, T phi, T m) :
        pt_(pt), eta_(eta), phi_(phi), m_(m) {}

    T pt() const { return pt_; }
    T eta() const { return eta_; }
    T phi() const { return phi_; }
    T m() const { return m_; }

    T px() const { return pt_ * cos(phi_); }
    T py() const { return pt_ * sin(phi_); }
    T theta() const { return atan(exp(-eta_)) * 2; }
    T pz() const { return pt_ / tan(theta()); }
    T e() const { sqrt(m_*m_ + pt_*pt_ + pz()*pz()); }

    // template <class T2> FourVector<T> operator+(const FourVector<T2> &other) const {
    //     // Dobbiamo usare gli accessors su other perché FourVector<T2> non è
    //     // lo stesso tipo di FourVector<T>
    //     return FourVector<T>(px_ + other.px(), py_ + other.py(),
    //                          pz_ + other.pz(), e_ + other.e());
    // }

    // Casting
    template <class T2> operator FourVectorPtEtaPhiM<T2>() const {
        return FourVectorPtEtaPhiM<T2>(pt_, eta_, phi_, m_);
    }

    operator FourVector<T>() const {
        return FourVector<T>(px(), py(), pz(), e());
    }

    FourVectorPtEtaPhiM<T> operator+(const FourVectorPtEtaPhiM<T> &other) const {
        return FourVectorPtEtaPhiM<T>(pt_ + other.pt_, TODO);
    }
  protected:
    T pt_, eta_, phi_, m_;
};

#endif // FOURVECTOR_H
