#!/usr/bin/env python3
import ROOT

ROOT.gInterpreter.ProcessLine('#include "simple_histo.C"')
ROOT.simple_histo()
