//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "veins/modules/application/traci/MyVeinsApp.h"

#include "veins/modules/application/traci/MyVeinsAppMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::MyVeinsApp);

void MyVeinsApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Start training the model with local data
        currentState = TRAINING;
        trainingRound = 0;

        vehicleId = mobility->getExternalId(); // Example flow0.2
        if (vehicleId.compare("flow0.0") == 0) {
            system("rm -rf weights logs data tmp");
            system("mkdir weights logs data tmp");
            py::initialize_interpreter();
        }
    } else if (stage == 1) {
        findHost()->getDisplayString().setTagArg("i", 1, "red");

        cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
        scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);

        cMessage* gossipModelMessage = new cMessage("Send (gossip) model to peers", GOSSIP_MODEL);
        scheduleAt(simTime() + GOSSIP_ROUND_TIME, gossipModelMessage);
    }
}

void MyVeinsApp::finish()
{
    DemoBaseApplLayer::finish();
    if (vehicleId.compare("flow0.0") == 0) {
        py::finalize_interpreter();
    }
}

bool MyVeinsApp::shouldAcceptMessageFromSender(std::string senderId)
{
    bool shouldAccept = true;
    for(auto it = lastReceivedVehicleIds.cbegin(); it != lastReceivedVehicleIds.cend(); it++) {
        if (senderId.compare(*it) == 0) {
            shouldAccept = false;
            break;
        }
    }
    if (shouldAccept) {
        if (lastReceivedVehicleIds.size() >= MAX_QUEUE_SIZE) {
            lastReceivedVehicleIds.pop_front();
        }
        lastReceivedVehicleIds.push_back(senderId);
    }
    return shouldAccept;
}

void MyVeinsApp::onWSM(BaseFrame1609_4* frame)
{
    MyVeinsAppMessage* wsm = check_and_cast<MyVeinsAppMessage*>(frame);

    std::cout << vehicleId << ", from " << wsm->getSenderId() << std::endl;
    if (currentState == WAITING) {
        if (shouldAcceptMessageFromSender(wsm->getSenderId())) {
            py::module_ simplenet = py::module_::import("simplenet");
            simplenet.attr("merge")(wsm->getWeights(), vehicleId);

            findHost()->getDisplayString().setTagArg("i", 1, "red");
            currentState = TRAINING;
            cMessage* trainingMessage = new cMessage("Training local model", LOCAL_TRAINING);
            scheduleAt(simTime() + TRAINING_TIME + uniform(0.0, 5.0), trainingMessage);
        } else {
            std::cerr << "onWSM - Received model ignored because the node have received a message from " << wsm->getSenderId() << " recently" << std::endl;
        }
    } else {
        std::cerr << "onWSM - Received model ignored because the node is already training" << std::endl;
    }
}

void MyVeinsApp::handleSelfMsg(cMessage* msg)
{
    std::cout << "Node " << vehicleId << ", action " << msg->getKind() << std::endl;

    switch (msg->getKind()) {
    case LOCAL_TRAINING: {
        trainingRound += 1;
        py::module_ simplenet = py::module_::import("simplenet");
        simplenet.attr("train")(vehicleId, trainingRound);

        findHost()->getDisplayString().setTagArg("i", 1, "green");
        currentState = WAITING;
        break;
    }
    case GOSSIP_MODEL: {
        py::module_ simplenet = py::module_::import("simplenet");
        py::str weights_py = simplenet.attr("get_weights")(vehicleId);
        std::string weights = weights_py;

        MyVeinsAppMessage* wsm = new MyVeinsAppMessage();
        wsm->setWeights(weights.c_str());
        wsm->setSenderAddress(myId);
        wsm->setSenderId(vehicleId.c_str());
        populateWSM(wsm);
        sendDelayedDown(wsm, uniform(0.0, 0.5));

        cMessage* gossipModelMessage = new cMessage("Send (gossip) model to peers", GOSSIP_MODEL);
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

void MyVeinsApp::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
}
