
#include "veins/modules/application/traci/GossipLearningApp.h"
#include "veins/modules/application/traci/AppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::GossipLearningApp);

void GossipLearningApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // TODO: implement here
        carId = mobility->getExternalId(); // Example flow0.2
        if (carId.compare("flow0.0") == 0) {
            system("rm -rf weights");
            system("mkdir weights");
            py::initialize_interpreter();
        }
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "red");
        // TODO: implement here
    }
}

void GossipLearningApp::finish()
{
    DemoBaseApplLayer::finish();
    // TODO: check car ID
    if (carId.compare("flow0.0") == 0) {
        py::finalize_interpreter();
    }
}

void GossipLearningApp::onWSM(BaseFrame1609_4* frame)
{
    AppMessage* wsm = check_and_cast<AppMessage*>(frame);
    // TODO: implement here
}

void GossipLearningApp::handleSelfMsg(cMessage* msg)
{
    // TODO: implement here
    cancelAndDelete(msg);
}