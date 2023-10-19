
#include "veins/modules/application/traci/GossipLearningApp.h"
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::GossipLearningApp);

void GossipLearningApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Start training the model with local data
        currentState = TRAINING;
        trainingRound = 0;

        vehicleId = mobility->getExternalId();
        if (vehicleId.compare("v0") == 0) {
            system("rm -rf weights logs");
            system("mkdir weights logs");
            py::initialize_interpreter();
        }
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "red");

        py::module_ learning = py::module_::import("learning");
        learning.attr("init")(vehicleId, simTime().dbl());

        cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);

        cMessage* gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME + uniform(0.0, 10.0), gossipModelMessage);
    }
}

void GossipLearningApp::finish()
{
    DemoBaseApplLayer::finish();
    if (vehicleId.compare("v0") == 0) {
        py::finalize_interpreter();
    }
}

void GossipLearningApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << vehicleId << " received message from " << wsm->getSenderId() << std::endl;
    if (currentState == WAITING) {
        EV << vehicleId << " merge models" << std::endl;

        py::module_ learning = py::module_::import("learning");
        learning.attr("merge")(wsm->getWeights(), vehicleId, wsm->getSenderId(), simTime().dbl());

        findHost()->getDisplayString().setTagArg("i", 1, "red");
        currentState = TRAINING;
        cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
    } else {
        EV_WARN << "onWSM - Received model ignored because the node is already training" << std::endl;
    }
}

void GossipLearningApp::handleSelfMsg(cMessage* msg)
{
    EV << "Node " << vehicleId << ", action " << msg->getKind() << std::endl;

    switch (msg->getKind()) {
    case LOCAL_TRAINING: {
        EV << "Node " << vehicleId << " local training" << std::endl;

        trainingRound += 1;
        py::module_ learning = py::module_::import("learning");
        learning.attr("train")(vehicleId, trainingRound, simTime().dbl());

        findHost()->getDisplayString().setTagArg("i", 1, "green");
        currentState = WAITING;
        break;
    }
    case GOSSIP_MODEL: {
        EV << "Node " << vehicleId << " gossip model" << std::endl;

        py::module_ learning = py::module_::import("learning");
        py::str weights_py = learning.attr("get_weights")(vehicleId, simTime().dbl());
        std::string weights = weights_py;

        AppMessage* wsm = new AppMessage();
        wsm->setWeights(weights.c_str());
        wsm->setSenderAddress(myId);
        wsm->setSenderId(vehicleId.c_str());
        populateWSM(wsm);
        sendDelayedDown(wsm, uniform(0.0, 0.5));

        cMessage* gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME, gossipModelMessage);
        break;
    }
    default: {
        EV_WARN << "handleSelfMsg - The message type was not detected" << std::endl;
        break;
    }
    }

    cancelAndDelete(msg);
}
