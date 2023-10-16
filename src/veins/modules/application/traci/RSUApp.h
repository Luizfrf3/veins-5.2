
#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayerRSU.h"

namespace veins {

class VEINS_API RSUApp : public DemoBaseApplLayerRSU {
public:
    void initialize(int stage) override;
protected:
    void onWSM(BaseFrame1609_4* wsm) override;
    void handleSelfMsg(cMessage* msg) override;
    void handleGateMsg(cMessage* msg) override;

    std::string rsuId;
    double posX;
    double posY;
    std::vector<double> rsuWeights;
private:
    double calculateDistance(double x1, double y1, double x2, double y2);
};

} // namespace veins
