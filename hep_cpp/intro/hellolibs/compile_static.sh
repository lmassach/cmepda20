#!/usr/bin/bash

# Compiliamo la libreria senza linkarla (position independet code, pic)
gcc -c -fpic hellolib.cpp -o hellolib.o
# Linkiamo qua
g++ main.cpp hellolib.o -o main_static.out
