#include <iostream>
#include "fourvector.h"
#include "particle.h"
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

Particle inputParticle() {
    double charge;
    input("Charge? ", charge);
    return Particle(charge, inputFourVector());
}

void printParticle(const Particle &p) {
    cout << "Particle of charge " << p.charge() << endl;
    printFourVector(p);
}

int main() {
    Particle p1 = inputParticle();
    cout << endl << endl;
    printParticle(p1);

    return 0;
}
