#ifndef __CLING__
#include "TH1.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TStyle.h"
#endif

namespace myns {
    void simple_histo() {
        gStyle->SetOptFit(111);
        gStyle->SetOptStat(0);
        TH1F *histo = new TH1F("histo", "Example histogram", 64, -8, 8);
        TCanvas *c = new TCanvas("c", "Example canvas", 600, 450);
        histo->SetBinErrorOption(TH1::EBinErrorOpt::kPoisson);
        histo->FillRandom("gaus");
        histo->Draw("E0");

        TF1 *ff = new TF1("ff", "gaus", -8, 8);
        ff->SetParameters(5000, 0, 1);
        histo->Fit(ff, "ILQR");

        c->SetLogy();
        c->SetGrid();
        c->SaveAs("simple_histo.png");
        c->Draw();
    }
}
