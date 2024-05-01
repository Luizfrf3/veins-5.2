
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <veins/modules/application/traci/WSCCApp.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::WSCCApp);

void WSCCApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Start training the model with local data
        currentState = WAITING;
        trainingRound = 0;

        vehicleId = mobility->getExternalId();
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "green");

        py::module_ learning = py::module_::import("learning");
        learning.attr("init")(vehicleId, simTime().dbl());
    }
}

void WSCCApp::finish()
{
    DemoBaseApplLayer::finish();
}

std::set<std::string> WSCCApp::splitString(std::string s, char del) {
    std::set<std::string> parts;
    std::stringstream ss(s);
    std::string word;
    while (!ss.eof()) {
        getline(ss, word, del);
        parts.insert(word);
    }
    return parts;
}

void WSCCApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    std::cout << vehicleId << " received message from " << wsm->getSenderId() << std::endl;
    if (currentState == WAITING && wsm->isRSU()) {
        std::set<std::string> participatingNodes = splitString(wsm->getParticipatingNodes(), ',');
        std::set<std::string> clusterNodes = splitString(wsm->getClusterNodes(), ',');
        if (clusterNodes.count(vehicleId) > 0 || (participatingNodes.count(vehicleId) == 0 && strcmp("global", wsm->getSenderId()) == 0)) {
            if (participatingNodes.count(vehicleId) == 0) {
                std::cout << vehicleId << " did not participate in the server aggregation" << std::endl;
            }
            std::cout << vehicleId << " started training" << std::endl;

            py::module_ learning = py::module_::import("learning");
            learning.attr("receive_global_model")(wsm->getWeights(), vehicleId, wsm->getSenderId(), simTime().dbl());

            findHost()->getDisplayString().setTagArg("i", 1, "red");
            currentState = TRAINING;
            cMessage* trainingMessage = new cMessage("Training local model");
            scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
        } else {
            std::cerr << "onWSM - Received model ignored because the vehicle belongs to another cluster" << std::endl;
        }
    } else {
        if (currentState == TRAINING) {
            std::cerr << "onWSM - Received model ignored because the node is already training" << std::endl;
        } else {
            std::cerr << "onWSM - Received model ignored because the message is from another vehicle" << std::endl;
        }
    }
}

void WSCCApp::handleSelfMsg(cMessage* msg)
{
    std::cout << "Node " << vehicleId << " ending training, round " << trainingRound << std::endl;

    py::module_ learning = py::module_::import("learning");
    learning.attr("train")(vehicleId, trainingRound, simTime().dbl());
    trainingRound += 1;

    findHost()->getDisplayString().setTagArg("i", 1, "green");
    currentState = WAITING;

    std::cout << "Node " << vehicleId << " sending model to server" << std::endl;

    py::str weights_py = learning.attr("get_weights")(vehicleId, simTime().dbl());
    std::string weights = weights_py;
    py::int_ dataset_size_py = learning.attr("get_dataset_size")(vehicleId);
    int datasetSize = dataset_size_py;

    AppMessage* wsm = new AppMessage();
    wsm->setWeights(weights.c_str());
    wsm->setDatasetSize(datasetSize);
    wsm->setSenderAddress(myId);
    wsm->setSenderId(vehicleId.c_str());
    populateWSM(wsm);
    sendDelayedDown(wsm, uniform(0.0, 0.5));

    cancelAndDelete(msg);
}
