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

#pragma once

#include "veins/veins.h"

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"

#include <queue>

using namespace omnetpp;

namespace veins {

/**
 * @brief
 * This is a stub for a typical Veins application layer.
 * Most common functions are overloaded.
 * See MyVeinsApp.cc for hints
 *
 * @author David Eckhoff
 *
 */

class VEINS_API MyVeinsApp : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    void finish() override;

protected:
    const int TRAINING_TIME = 10;
    const int GOSSIP_ROUND_TIME = 20;
    const int MAX_QUEUE_SIZE = 2;

    enum nodeState {
        TRAINING,
        WAITING
    };
    enum selfMessageKind {
        LOCAL_TRAINING,
        GOSSIP_MODEL
    };

    nodeState currentState;
    std::string vehicleId;
    int trainingRound;
    std::deque<std::string> lastReceivedVehicleIds;

protected:
    void onWSM(BaseFrame1609_4* frame) override;

    void handleSelfMsg(cMessage* msg) override;
    void handlePositionUpdate(cObject* obj) override;

private:
    bool shouldAcceptMessageFromSender(std::string senderId);
};

} // namespace veins
