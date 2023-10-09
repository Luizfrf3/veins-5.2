
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

        carId = mobility->getExternalId();
        if (carId.compare("v0") == 0) {
            system("rm -rf weights");
            system("mkdir weights");
            py::initialize_interpreter();
        }
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "red");

        py::module_ learning = py::module_::import("learning");
        learning.attr("init")(carId);

        cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);

        cMessage* gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME + uniform(0.0, 10.0), gossipModelMessage);
    }
}

void GossipLearningApp::finish()
{
    DemoBaseApplLayer::finish();
    if (carId.compare("v0") == 0) {
        py::finalize_interpreter();
    }
}

void GossipLearningApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << carId << " received message from " << wsm->getSenderId() << std::endl;
    if (currentState == WAITING) {
        EV << carId << " merge models" << std::endl;

        py::module_ learning = py::module_::import("learning");
        learning.attr("merge")(wsm->getWeights(), carId);

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
    EV << "Node " << carId << ", action " << msg->getKind() << std::endl;

    switch (msg->getKind()) {
    case LOCAL_TRAINING: {
        EV << "Node " << carId << " local training" << std::endl;

        trainingRound += 1;
        py::module_ learning = py::module_::import("learning");
        learning.attr("train")(carId, trainingRound);

        findHost()->getDisplayString().setTagArg("i", 1, "green");
        currentState = WAITING;
        break;
    }
    case GOSSIP_MODEL: {
        EV << "Node " << carId << " gossip model" << std::endl;

        py::module_ learning = py::module_::import("learning");
        py::str weights_py = learning.attr("get_weights")(carId);
        std::string weights = weights_py;

        AppMessage* wsm = new AppMessage();
        wsm->setWeights(weights.c_str());
        wsm->setSenderAddress(myId);
        wsm->setSenderId(carId.c_str());
        populateWSM(wsm);
        sendDelayedDown(wsm, uniform(0.0, 1.0));

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
