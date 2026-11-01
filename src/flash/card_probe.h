#pragma once
// Per-card calibration (heterogeneous cards) and the slow-card test knob; card_probe.cu.
#include <string>
#include <vector>

namespace flash {
struct CardCal {
    int dev = -1, sms = 0;
    bool ok = false;
    double tflops = 0, gbs = 0;   // synthetic FP16 FMA throughput, streaming read bandwidth
    double vram = 0, free = 0;    // bytes
    char name[64] = "";
};
CardCal calibrate_card(int dev, int scale = 1);  // scale: longer kernels, more repeats (re-check)
int test_slow_cards();            // FLASH_TEST_SLOW_CARD=<ordinal>:<fraction of SMs held>[,...] (tests only)
std::string card_line(const std::vector<CardCal>& c);
// build, cards (links, NUMA, host rate), per-pair peer access and copy rates: printed on 4+ card and PP x TP boots and
// with FLASH_BOX_REPORT=1 (remote diagnosis)
std::string box_report(const std::vector<int>& devs);
}  // namespace flash
