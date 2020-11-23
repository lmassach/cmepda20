#include <iostream>
#include "hellolib.h"
// Il secondo include non è fondamentale se non implementiamo classi, ma
// in questo modo abbiamo un controllo di consistenza

void printSomething(const char *what) {
    std::cout << what << std::endl;
}
