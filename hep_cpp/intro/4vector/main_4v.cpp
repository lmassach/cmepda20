#include <iostream>
#include "fourvector.h"
using namespace std;

const double pi = 3.14159265359;
const double deg = pi / 180.0;

inline void input(const char* prompt, double &var) {
    cout << prompt;
    cin >> var;
}

FourVector inputFourVector() {
    double px, py, pz, e;
    input("Px? ", px);
    input("Py? ", py);
    input("Pz? ", pz);
    input("E? ", e);
    return FourVector(px, py, pz, e);
}

void printFourVector(const FourVector &v) {
    cout << "Px = " << v.px() << endl;
    cout << "Py = " << v.py() << endl;
    cout << "Pz = " << v.pz() << endl;
    cout << "E = " << v.e() << endl << endl;

    cout << "M = " << v.m() << endl;
    cout << "Pt = " << v.pt() << endl;
    cout << "P = " << v.p() << endl << endl;

    cout << "phi = " << v.phi() / deg << "°" << endl;
    cout << "theta = " << v.theta() / deg << "°" << endl << endl;

    cout << "y = " << v.y() << endl;
    cout << "eta = " << v.eta() << endl << endl;

    cout << "gamma = " << v.gamma() << endl;
    cout << "beta = " << v.beta() << endl << endl << endl;
}

int main() {
    FourVector v1 = inputFourVector();
    printFourVector(v1);

    FourVector v2 = inputFourVector();
    printFourVector(v2);

    cout << "Sum" << endl;
    FourVector v3 = v1 + v2;
    printFourVector(v3);

    return 0;
}
