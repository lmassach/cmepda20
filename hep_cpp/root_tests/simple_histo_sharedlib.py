#!/usr/bin/env python3
import subprocess
import ROOT

subprocess.run("g++ -fpic -shared simple_histo.C -o simple_histo.so "
               "`root-config --cflags --libs`", shell=True, check=True)

ROOT.gInterpreter.Declare("void simple_histo();")
ROOT.gInterpreter.Load("./simple_histo.so")
ROOT.simple_histo()
