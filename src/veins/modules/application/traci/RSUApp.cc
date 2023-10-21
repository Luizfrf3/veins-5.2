
#include "veins/modules/application/traci/RSUApp.h"
#include "veins/modules/application/traci/AppMessage_m.h"

#include <random>

using namespace veins;

Define_Module(veins::RSUApp);

double RSUApp::calculateDistance(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void RSUApp::initialize(int stage)
{
    DemoBaseApplLayerRSU::initialize(stage);

    if (stage == 0) {
        rsuId = std::to_string(getParentModule()->getIndex());
        gen.seed(SEED);
    } else if (stage == 1) {
        int numberOfRSUs = getSimulation()->getModuleByPath("rsu[*]")->getVectorSize();

        std::vector<double> xs(numberOfRSUs);
        std::vector<double> ys(numberOfRSUs);
        for (int i = 0; i < numberOfRSUs; i++) {
            std::string id = std::to_string(i);
            std::string path = "rsu[" + id + "].mobility";
            double x = check_and_cast<BaseMobility*>(getSimulation()->getModuleByPath(path.c_str()))->getPositionAt(simTime()).x;
            double y = check_and_cast<BaseMobility*>(getSimulation()->getModuleByPath(path.c_str()))->getPositionAt(simTime()).y;
            if (stoi(rsuId) == i) {
                posX = x;
                posY = y;
            }
            xs[i] = x;
            ys[i] = y;
        }

        for (int i = 0; i < numberOfRSUs; i++) {
            // Until 1000 meters the weight is 1, after that it increases +1 every 500 meters
            double distance = calculateDistance(posX, posY, xs[i], ys[i]);
            if (stoi(rsuId) == i) {
                // So that the weight is 0 for the current RSU
                distance = 500;
            } else if (distance < 1000) {
                distance = 1000;
            }
            double weight = ceil((distance - 1000) / 500) + 1;
            rsuWeights.push_back(weight);
        }

        uniform_dist.param(std::uniform_int_distribution<>::param_type(0, numberOfRSUs - 1));
        weighted_dist.param(std::discrete_distribution<>::param_type(std::begin(rsuWeights), std::end(rsuWeights)));

        EV << "ID: " << rsuId << ", X: " << posX << ", Y: " << posY << std::endl;
    }
}

void RSUApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);

    EV << "RSU " << rsuId << " received message from " << wsm->getSenderId() << std::endl;
    if (wsm->isRSU() == false) {
        int dest;
        do {
            if (messageStrategy == RANDOM) {
                dest = uniform_dist(gen);
            } else {
                dest = weighted_dist(gen);
            }
        } while (dest == stoi(rsuId));

        AppMessage* msg = new AppMessage();
        msg->setWeights(wsm->getWeights());
        msg->setDatasetSize(wsm->getDatasetSize());
        msg->setSenderAddress(wsm->getSenderAddress());
        msg->setSenderId(wsm->getSenderId());
        msg->setDest(dest);
        send(msg, "out");
    } else {
        EV_WARN << "onWSM - Received model ignored because it is from another RSU" << std::endl;
    }
}

void RSUApp::handleGateMsg(cMessage* msg)
{
    AppMessage* appMsg = check_and_cast<AppMessage*>(msg);

    AppMessage* wsm = new AppMessage();
    wsm->setWeights(appMsg->getWeights());
    wsm->setDatasetSize(appMsg->getDatasetSize());
    wsm->setSenderAddress(appMsg->getSenderAddress());
    wsm->setSenderId(appMsg->getSenderId());
    wsm->setIsRSU(true);
    populateWSM(wsm);
    sendDelayedDown(wsm, uniform(0.0, 0.5));

    delete(msg);
}
