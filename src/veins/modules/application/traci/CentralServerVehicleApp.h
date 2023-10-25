
#pragma once

#include "veins/veins.h"
#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"

using namespace omnetpp;

namespace veins {

class VEINS_API CentralServerVehicleApp : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    void finish() override;

protected:
    const int TRAINING_TIME = 15;

    enum nodeState {
        TRAINING,
        WAITING
    };

    nodeState currentState;
    std::string vehicleId;
    int trainingRound;

    void onWSM(BaseFrame1609_4* frame) override;
    void handleSelfMsg(cMessage* msg) override;
};

} // namespace veins
