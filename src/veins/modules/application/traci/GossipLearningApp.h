
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
    const int TRAINING_TIME = 10;
    const int GOSSIP_ROUND_TIME = 30;

    enum nodeState {
        TRAINING,
        WAITING
    };
    enum selfMessageKind {
        LOCAL_TRAINING,
        GOSSIP_MODEL
    };

    nodeState currentState;
    std::string carId;
    int trainingRound;

    void onWSM(BaseFrame1609_4* frame) override;
    void handleSelfMsg(cMessage* msg) override;
};

} // namespace veins
