#include <iostream>
#include "phonebook.h"
using namespace std;

int main() {
    PhoneBookEntry pbe("Aleph", "B", "0");
    cout << pbe.strFull() << endl << endl;
    PhoneBookVOC pb;
    pb.addEntry(pbe);
    cout << pb.str() << endl << endl;

    pbe.name = "Alpha";
    pb.addEntry("AA", "BB", "+391");
    pb.at(1).addNumber("work", "12");

    PhoneBookEntry pbe2("A", "B", "10");
    pbe2.addNumber("work", "22");
    cout << pbe2.strFull() << endl;
    pb.addEntry(pbe2);

    pb.addPrefixes();
    pb.sort();
    cout << pbe.strFull() << endl << endl;
    cout << pb.str() << endl << endl;

    return 0;
}
