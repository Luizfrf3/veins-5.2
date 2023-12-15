
#include "veins/modules/application/traci/OurMethodApp.h"
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::OurMethodApp);

void OurMethodApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Start training the model with local data
        currentState = TRAINING;
        trainingRound = 0;
        numberOfReceivedModels = 0;

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

void OurMethodApp::finish()
{
    DemoBaseApplLayer::finish();
    if (vehicleId.compare("v0") == 0) {
        py::finalize_interpreter();
    }
}

void OurMethodApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << vehicleId << " received message from " << wsm->getSenderId() << ", RSU " << wsm->isRSU() << std::endl;
    if (wsm->getSenderAddress() == myId) {
        EV_WARN << "onWSM - Received model ignored because it is from the same node" << std::endl;
    } else if (currentState == WAITING) {
        EV << vehicleId << " store model" << std::endl;

        numberOfReceivedModels += 1;
        py::module_ learning = py::module_::import("learning");
        learning.attr("store_weights")(wsm->getWeights(), wsm->getDatasetSize(), vehicleId, wsm->getSenderId(), simTime().dbl());
    } else {
        EV_WARN << "onWSM - Received model ignored because the node is already training" << std::endl;
    }
}

void OurMethodApp::handleSelfMsg(cMessage* msg)
{
    EV << "Node " << vehicleId << ", action " << msg->getKind() << std::endl;

    switch (msg->getKind()) {
    case LOCAL_TRAINING: {
        EV << "Node " << vehicleId << " ending training, round " << trainingRound << ", received models " << numberOfReceivedModels << std::endl;

        py::module_ learning = py::module_::import("learning");
        learning.attr("train")(vehicleId, trainingRound, simTime().dbl());
        trainingRound += 1;

        findHost()->getDisplayString().setTagArg("i", 1, "green");
        currentState = WAITING;
        numberOfReceivedModels = 0;

        EV << "Node " << vehicleId << " gossiping model" << std::endl;

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

        break;
    }
    case GOSSIP_MODEL: {
        EV << "Node " << vehicleId << " gossip model" << std::endl;

        if (numberOfReceivedModels > 0) {
            EV << "Node " << vehicleId << " started training, round " << trainingRound << std::endl;

            findHost()->getDisplayString().setTagArg("i", 1, "red");
            currentState = TRAINING;
            cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
            scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
        } else {
            EV << "Node " << vehicleId << " gossiping model" << std::endl;

            py::module_ learning = py::module_::import("learning");
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
        }

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
