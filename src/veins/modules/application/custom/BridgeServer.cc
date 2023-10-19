
#include "veins/modules/application/traci/AppMessage_m.h"

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>

namespace omnetpp {

class BridgeServer : public cSimpleModule {
protected:
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(BridgeServer);

void BridgeServer::handleMessage(cMessage *msg)
{
    veins::AppMessage* appMsg = check_and_cast<veins::AppMessage*>(msg);
    EV << "Message to " << appMsg->getDest() << " arrived in Bridge Server." << std::endl;
    send(msg, "gate$o", appMsg->getDest());
}

}
