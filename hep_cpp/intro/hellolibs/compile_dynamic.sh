#!/usr/bin/bash

# Compiliamo la libreria senza linkarla (position independet code, pic)
gcc -shared -fPIC -o libhellolib.so hellolib.cpp
# Linkiamo qua
g++ main.cpp -o main_dynamic.out -L. -lhellolib

# Run with LD_LIBRARY_PATH=. ./main_dynamic.out
# Also try LD_LIBRARY_PATH=. ldd ./main_dynamic.out

# If it throws errors with unreadable symbols, try
#   echo <unreadable_symbol> | c++filt
# to get a readable thing.
