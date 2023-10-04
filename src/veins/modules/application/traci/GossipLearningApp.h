
#pragma once

#include "veins/veins.h"
#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"

using namespace omnetpp;

namespace veins {

class VEINS_API GossipLearningApp : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    void finish() override;

protected:
    std::string carId;

    void onWSM(BaseFrame1609_4* frame) override;
    void handleSelfMsg(cMessage* msg) override;
};

} // namespace veins
