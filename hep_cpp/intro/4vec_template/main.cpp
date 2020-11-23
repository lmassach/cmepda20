#include <iostream>
#include "fourvector.h"
using namespace std;

const double pi = 3.14159265359;
const double deg = pi / 180.0;

inline void input(const char* prompt, double &var) {
    cout << prompt;
    cin >> var;
}

template <class T> FourVector<T> inputFourVector() {
    double px, py, pz, e;
    input("Px? ", px);
    input("Py? ", py);
    input("Pz? ", pz);
    input("E? ", e);
    return FourVector<T>(px, py, pz, e);
}

template <class T> void printFourVector(const FourVector<T> &v) {
    cout << "(px, py, pz, E) = (" << v.px() << ", " << v.py() << ", "
         << v.pz() << ", " << v.e() << ");  "
         << "(pt, eta, phi, m) = (" << v.pt() << ", " << v.eta() << ", "
         << v.phi() / deg << "°, " << v.m() << ")" << endl;
}

int main() {
    FourVector<double> v1 = inputFourVector<double>();
    printFourVector(v1);
    cout << endl;

    FourVector<float> v2 = inputFourVector<float>();
    printFourVector(v2);
    cout << endl;

    cout << "Sum" << endl;
    FourVector<double> v3 = v1 + v2;
    printFourVector(v3);

    return 0;
}
