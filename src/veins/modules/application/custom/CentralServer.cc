
#include "veins/modules/application/traci/AppMessage_m.h"

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace omnetpp {

class CentralServer : public cSimpleModule {
public:
    virtual ~CentralServer();
protected:
    const int ROUND_TIME = 25;
    const int AGGREGATION_TIME = 5;

    std::string SERVER = "server";

    enum nodeState {
        AGGREGATING,
        WAITING
    };
    enum selfMessageKind {
        ROUND_MESSAGE,
        AGGREGATION_MESSAGE
    };

    nodeState currentState;
    int numberOfRSUs;
    int aggregationRound;
    int numberOfReceivedModels;

    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(CentralServer);

CentralServer::~CentralServer()
{
    py::finalize_interpreter();
}

void CentralServer::initialize()
{
    system("rm -rf weights logs");
    system("mkdir weights logs");
    py::initialize_interpreter();

    numberOfRSUs = getSimulation()->getModuleByPath("rsu[*]")->getVectorSize();
    aggregationRound = 0;
    numberOfReceivedModels = 0;
    currentState = AGGREGATING;
    cMessage *selfMsg = new cMessage("Initialization message", ROUND_MESSAGE);
    scheduleAt(ROUND_TIME, selfMsg);
}

void CentralServer::handleMessage(cMessage *msg)
{
    switch (msg->getKind()) {
    case ROUND_MESSAGE: {
        EV << SERVER << " starting model aggregation" << std::endl;

        currentState = AGGREGATING;
        cMessage *selfMsg = new cMessage("Aggregating models", AGGREGATION_MESSAGE);
        scheduleAt(AGGREGATION_TIME, selfMsg);
        break;
    }
    case AGGREGATION_MESSAGE: {
        EV << SERVER << " ending aggregation round " << aggregationRound << ", received models" << numberOfReceivedModels << std::endl;

        py::module_ learning = py::module_::import("learning");
        learning.attr("aggregation")(aggregationRound, simTime().dbl());
        aggregationRound += 1;
        numberOfReceivedModels = 0;

        currentState = WAITING;
        cMessage *selfMsg = new cMessage("Waiting models", ROUND_MESSAGE);
        scheduleAt(ROUND_TIME, selfMsg);

        py::str weights_py = learning.attr("get_weights")(SERVER, simTime().dbl());
        std::string weights = weights_py;
        veins::AppMessage* appMsg = new veins::AppMessage();
        appMsg->setWeights(weights.c_str());
        appMsg->setSenderId(SERVER.c_str());
        for (int i = 0; i < numberOfRSUs; i++) {
            send(appMsg, "gate$o", i);
        }

        break;
    }
    default: {
        veins::AppMessage* appMsg = check_and_cast<veins::AppMessage*>(msg);
        EV << "Central Server received message from " << appMsg->getSenderId() << std::endl;

        if (currentState == WAITING) {
            EV << SERVER << " store model" << std::endl;

            numberOfReceivedModels += 1;
            py::module_ learning = py::module_::import("learning");
            learning.attr("store_weights")(appMsg->getWeights(), appMsg->getDatasetSize(), SERVER, appMsg->getSenderId(), simTime().dbl());
        } else {
            EV_WARN << "handleMessage - Received model ignored because the server is already aggregating" << std::endl;
        }
        break;
    }
    }

    cancelAndDelete(msg);
}

}
