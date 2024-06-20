
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <veins/modules/application/traci/HybridMethodApp.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::HybridMethodApp);

void HybridMethodApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Start training the model with local data
        currentState = WAITING;
        trainingRound = 0;
        numberOfReceivedModels = 0;
        numberOfReceivedModelsWhileTraining = 0;
        receivedModelFromServer = false;

        vehicleId = mobility->getExternalId();
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "red");

        py::module_ learning = py::module_::import("learning");
        learning.attr("init")(vehicleId, simTime().dbl());

        trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);

        gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME + uniform(0.0, 10.0), gossipModelMessage);
    }
}

void HybridMethodApp::finish()
{
    DemoBaseApplLayer::finish();
}

std::set<std::string> HybridMethodApp::splitString(std::string s, char del) {
    std::set<std::string> parts;
    std::stringstream ss(s);
    std::string word;
    while (!ss.eof()) {
        getline(ss, word, del);
        parts.insert(word);
    }
    return parts;
}

void HybridMethodApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    std::cout << vehicleId << " received message from " << wsm->getSenderId() << std::endl;
    if (wsm->isRSU()) {
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
            receivedModelFromServer = true;
            numberOfReceivedModels = 0;
            numberOfReceivedModelsWhileTraining = 0;

            cancelAndDelete(gossipModelMessage);
            cancelAndDelete(trainingMessage);
            trainingMessage = new cMessage("Training local model");
            scheduleAt(simTime() + TRAINING_TIME - 3 + uniform(0.0, 5.0), trainingMessage);
        } else {
            std::cerr << "onWSM - Received model ignored because the vehicle belongs to another cluster" << std::endl;
        }
    } else {
        if (currentState == WAITING) {
            std::cout << vehicleId << " store model" << std::endl;

            numberOfReceivedModels += 1;
            py::module_ learning = py::module_::import("learning");
            learning.attr("store_weights")(wsm->getWeights(), wsm->getDatasetSize(), vehicleId, wsm->getSenderId(), simTime().dbl());
        } else {
            std::cout << vehicleId << " store model while training" << std::endl;

            numberOfReceivedModelsWhileTraining += 1;
            py::module_ learning = py::module_::import("learning");
            learning.attr("store_weights_while_training")(wsm->getWeights(), wsm->getDatasetSize(), vehicleId, wsm->getSenderId(), simTime().dbl());
        }
    }
}

void HybridMethodApp::handleSelfMsg(cMessage* msg)
{
    std::cout << "Node " << vehicleId << ", action " << msg->getKind() << std::endl;

    switch (msg->getKind()) {
    case LOCAL_TRAINING: {
        std::cout << "Node " << vehicleId << " ending training, round " << trainingRound << ", received models " << numberOfReceivedModels << std::endl;

        if (receivedModelFromServer == true) {
            gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
            scheduleAt(simTime() + GOSSIP_ROUND_TIME, gossipModelMessage);
            receivedModelFromServer = false;
        }

        py::module_ learning = py::module_::import("learning");
        learning.attr("train")(vehicleId, trainingRound, simTime().dbl());
        trainingRound += 1;

        findHost()->getDisplayString().setTagArg("i", 1, "green");
        currentState = WAITING;
        numberOfReceivedModels = numberOfReceivedModelsWhileTraining;
        numberOfReceivedModelsWhileTraining = 0;

        std::cout << "Node " << vehicleId << " gossiping or sending model to server" << std::endl;

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
        std::cout << "Node " << vehicleId << " gossip model" << std::endl;

        if (numberOfReceivedModels > 0) {
            std::cout << "Node " << vehicleId << " started training, round " << trainingRound << std::endl;

            findHost()->getDisplayString().setTagArg("i", 1, "red");
            currentState = TRAINING;
            trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
            scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
        } else {
            std::cout << "Node " << vehicleId << " gossiping model" << std::endl;

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

        gossipModelMessage = new cMessage("Send model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME, gossipModelMessage);
        break;
    }
    default: {
        std::cerr << "handleSelfMsg - The message type was not detected" << std::endl;
        break;
    }
    }

    cancelAndDelete(msg);
}
