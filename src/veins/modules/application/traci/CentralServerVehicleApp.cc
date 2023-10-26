
#include "veins/modules/application/traci/CentralServerVehicleApp.h"
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::CentralServerVehicleApp);

void CentralServerVehicleApp::initialize(int stage)
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

void CentralServerVehicleApp::finish()
{
    DemoBaseApplLayer::finish();
}

void CentralServerVehicleApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << vehicleId << " received message from " << wsm->getSenderId() << std::endl;
    if (currentState == WAITING && wsm->isRSU()) {
        EV << vehicleId << " started training" << std::endl;

        py::module_ learning = py::module_::import("learning");
        learning.attr("receive_global_model")(wsm->getWeights(), vehicleId, wsm->getSenderId(), simTime().dbl());

        findHost()->getDisplayString().setTagArg("i", 1, "red");
        currentState = TRAINING;
        cMessage* trainingMessage = new cMessage("Training local model");
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
    } else {
        if (currentState == WAITING) {
            EV_WARN << "onWSM - Received model ignored because the node is already training" << std::endl;
        } else {
            EV_WARN << "onWSM - Received model ignored because the message is from another vehicle" << std::endl;
        }
    }
}

void CentralServerVehicleApp::handleSelfMsg(cMessage* msg)
{
    EV << "Node " << vehicleId << " ending training, round " << trainingRound << std::endl;

    py::module_ learning = py::module_::import("learning");
    learning.attr("train")(vehicleId, trainingRound, simTime().dbl());
    trainingRound += 1;

    findHost()->getDisplayString().setTagArg("i", 1, "green");
    currentState = WAITING;

    EV << "Node " << vehicleId << " sending model to server" << std::endl;

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
