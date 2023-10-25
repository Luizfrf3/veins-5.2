
#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayerRSU.h"

#include <random>

namespace veins {

class VEINS_API OurMethodRSUApp : public DemoBaseApplLayerRSU {
public:
    void initialize(int stage) override;
protected:
    void onWSM(BaseFrame1609_4* wsm) override;
    void handleGateMsg(cMessage* msg) override;

    int SEED = 12;

    enum messageDisseminationStrategy {
        RANDOM,
        DISTANCE
    };
    messageDisseminationStrategy messageStrategy = RANDOM;

    std::mt19937 gen;
    std::uniform_int_distribution<int> uniform_dist;
    std::discrete_distribution<int> weighted_dist;

    std::string rsuId;
    double posX;
    double posY;
    std::vector<double> rsuWeights;
private:
    double calculateDistance(double x1, double y1, double x2, double y2);
};

} // namespace veins
