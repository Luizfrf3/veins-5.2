
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
    // TODO: implement here
}

}
