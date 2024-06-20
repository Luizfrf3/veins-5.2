
#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayerRSU.h"

#include <random>

namespace veins {

class VEINS_API HybridMethodRSUApp : public DemoBaseApplLayerRSU {
public:
    void initialize(int stage) override;
protected:
    void onWSM(BaseFrame1609_4* wsm) override;
    void handleGateMsg(cMessage* msg) override;

    std::string rsuId;
};

} // namespace veins
