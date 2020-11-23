#!/bin/bash
# No need to make an actual library for this example
g++ main_4v.cpp fourvector.cpp -o main_4v.out -lm
g++ main_part.cpp fourvector.cpp particle.cpp -o main_part.out -lm
