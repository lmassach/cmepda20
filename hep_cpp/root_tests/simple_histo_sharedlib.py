#!/usr/bin/env python3
import subprocess
import ROOT

subprocess.run("g++ -fpic -shared simple_histo.C -o simple_histo.so "
               "`root-config --cflags --libs`", shell=True, check=True)

# "Header" for the library. Also gInterpreter.Declare("code") can be used.
ROOT.gInterpreter.ProcessLine("namespace myns { void simple_histo(); }")
ROOT.gInterpreter.Load("./simple_histo.so")
ROOT.myns.simple_histo()
