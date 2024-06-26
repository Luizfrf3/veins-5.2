
#pragma once

#include "veins/veins.h"
#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"

using namespace omnetpp;

namespace veins {

class VEINS_API HybridMethodApp : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    void finish() override;

protected:
    const int TRAINING_TIME = 15;
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
    std::string vehicleId;
    int trainingRound;
    int numberOfReceivedModels;
    int numberOfReceivedModelsWhileTraining;
    bool receivedModelFromServer;
    cMessage* trainingMessage;
    cMessage* gossipModelMessage;

    void onWSM(BaseFrame1609_4* frame) override;
    void handleSelfMsg(cMessage* msg) override;
private:
    std::set<std::string> splitString(std::string s, char del);
};

} // namespace veins
