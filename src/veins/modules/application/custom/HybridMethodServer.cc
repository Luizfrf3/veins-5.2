
#include "veins/modules/application/traci/AppMessage_m.h"

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace omnetpp {

class HybridMethodServer : public cSimpleModule {
public:
    virtual ~HybridMethodServer();
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

Define_Module(HybridMethodServer);

HybridMethodServer::~HybridMethodServer()
{
    py::finalize_interpreter();
}

void HybridMethodServer::initialize()
{
    system("rm -rf weights logs data tmp");
    system("mkdir weights logs data tmp");
    py::initialize_interpreter();

    numberOfRSUs = getSimulation()->getModuleByPath("rsu[*]")->getVectorSize();
    aggregationRound = 0;
    numberOfReceivedModels = 0;
    currentState = AGGREGATING;

    py::module_ learning = py::module_::import("learning");
    learning.attr("init_server")(SERVER, simTime().dbl());

    cMessage *selfMsg = new cMessage("Waiting models", ROUND_MESSAGE);
    scheduleAt(simTime() + ROUND_TIME, selfMsg);
}

void HybridMethodServer::handleMessage(cMessage *msg)
{
    if (strcmp("Waiting models", msg->getName()) == 0) {
        std::cout << SERVER << " starting model aggregation" << std::endl;

        currentState = AGGREGATING;
        cMessage *selfMsg = new cMessage("Aggregating models", AGGREGATION_MESSAGE);
        scheduleAt(simTime() + AGGREGATION_TIME, selfMsg);
    } else if (strcmp("Aggregating models", msg->getName()) == 0) {
        std::cout << SERVER << " ending aggregation round " << aggregationRound << ", received models " << numberOfReceivedModels << std::endl;

        py::module_ learning = py::module_::import("learning");
        py::int_ number_of_clusters_py = learning.attr("aggregation")(aggregationRound, SERVER, simTime().dbl());
        int numberOfClusters = number_of_clusters_py;
        aggregationRound += 1;
        numberOfReceivedModels = 0;

        currentState = WAITING;
        cMessage *selfMsg = new cMessage("Waiting models", ROUND_MESSAGE);
        scheduleAt(simTime() + ROUND_TIME, selfMsg);

        py::str participating_nodes_py = learning.attr("get_participating_nodes")(SERVER, simTime().dbl());
        std::string participatingNodes = participating_nodes_py;

        if (numberOfClusters == 0) {
            std::cout << "Number of clusters is 0, sending global model" << std::endl;

            py::str weights_py = learning.attr("get_weights")(SERVER, simTime().dbl());
            std::string weights = weights_py;
            for (int i = 0; i < numberOfRSUs; i++) {
                veins::AppMessage* appMsg = new veins::AppMessage();
                appMsg->setWeights(weights.c_str());
                appMsg->setSenderId("global");
                appMsg->setParticipatingNodes("");
                appMsg->setClusterNodes("");
                send(appMsg, "gate$o", i);
            }
        } else {
            std::cout << "Sending models to " << numberOfClusters << " clusters, participating nodes " << participatingNodes << std::endl;
            for (int i = -1; i < numberOfClusters; i++) {
                py::str weights_py = learning.attr("get_cluster_weights")(SERVER, i, simTime().dbl());
                std::string weights = weights_py;
                py::str cluster_nodes_py = learning.attr("get_cluster_nodes")(SERVER, i, simTime().dbl());
                std::string clusterNodes = cluster_nodes_py;
                std::cout << "Sending model to cluster " << i << ", cluster nodes " << clusterNodes << std::endl;
                for (int j = 0; j < numberOfRSUs; j++) {
                    veins::AppMessage* appMsg = new veins::AppMessage();
                    appMsg->setWeights(weights.c_str());
                    if (i == -1) {
                        appMsg->setSenderId("global");
                    } else {
                        appMsg->setSenderId(i);
                    }
                    appMsg->setParticipatingNodes(participatingNodes.c_str());
                    appMsg->setClusterNodes(clusterNodes.c_str());
                    send(appMsg, "gate$o", j);
                }
            }
        }
    } else {
        veins::AppMessage* appMsg = check_and_cast<veins::AppMessage*>(msg);
        std::cout << "Central Server received message from " << appMsg->getSenderId() << std::endl;

        if (currentState == WAITING) {
            std::cout << SERVER << " store model" << std::endl;

            numberOfReceivedModels += 1;
            py::module_ learning = py::module_::import("learning");
            learning.attr("store_weights")(appMsg->getWeights(), appMsg->getDatasetSize(), SERVER, appMsg->getSenderId(), simTime().dbl());
        } else {
            std::cerr << "handleMessage - Received model ignored because the server is already aggregating" << std::endl;
        }
    }

    cancelAndDelete(msg);
}

}
