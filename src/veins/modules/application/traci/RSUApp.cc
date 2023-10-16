
#include "veins/modules/application/traci/RSUApp.h"

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
                distance = 0;
            } else if (distance < 1000) {
                distance = 1000;
            }
            double weight = ceil((distance - 1000) / 500) + 1;
            rsuWeights.push_back(weight);
        }

        EV << "ID: " << rsuId << ", X: " << posX << ", Y: " << posY << std::endl;
    }
}

void RSUApp::onWSM(BaseFrame1609_4* frame)
{
    // TODO: implement here
    //std::mt19937 gen(12);
    //std::uniform_int_distribution<> distr(0, numberOfRSUs - 1);
    //std::vector<double> weights{90,56,10};
    //std::discrete_distribution<int> distr(std::begin(weights), std::end(weights));
    //EV << distr(gen) << std::endl;
}

void RSUApp::handleSelfMsg(cMessage* msg)
{
    // TODO: implement here
    cancelAndDelete(msg);
}

void RSUApp::handleGateMsg(cMessage* msg)
{
    // TODO: implement here
    delete(msg);
}
