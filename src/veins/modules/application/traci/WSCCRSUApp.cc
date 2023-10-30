
#include <veins/modules/application/traci/WSCCRSUApp.h>
#include "veins/modules/application/traci/AppMessage_m.h"

#include <random>

using namespace veins;

Define_Module(veins::WSCCRSUApp);

void WSCCRSUApp::initialize(int stage)
{
    DemoBaseApplLayerRSU::initialize(stage);

    if (stage == 0) {
        rsuId = std::to_string(getParentModule()->getIndex());
    }
}

void WSCCRSUApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << "RSU " << rsuId << " received message from " << wsm->getSenderId() << std::endl;
    if (wsm->isRSU() == false) {
        AppMessage* msg = new AppMessage();
        msg->setWeights(wsm->getWeights());
        msg->setDatasetSize(wsm->getDatasetSize());
        msg->setSenderAddress(wsm->getSenderAddress());
        msg->setSenderId(wsm->getSenderId());
        send(msg, "out");
    } else {
        EV_WARN << "onWSM - Received model ignored because it is from another RSU" << std::endl;
    }
}

void WSCCRSUApp::handleGateMsg(cMessage* msg)
{
    AppMessage* appMsg = check_and_cast<AppMessage*>(msg);

    AppMessage* wsm = new AppMessage();
    wsm->setWeights(appMsg->getWeights());
    wsm->setDatasetSize(appMsg->getDatasetSize());
    wsm->setSenderAddress(appMsg->getSenderAddress());
    wsm->setSenderId(appMsg->getSenderId());
    wsm->setIsRSU(true);
    wsm->setParticipatingNodes(appMsg->getParticipatingNodes());
    wsm->setClusterNodes(appMsg->getClusterNodes());
    populateWSM(wsm);
    sendDelayedDown(wsm, uniform(0.0, 0.5));

    cancelAndDelete(msg);
}
