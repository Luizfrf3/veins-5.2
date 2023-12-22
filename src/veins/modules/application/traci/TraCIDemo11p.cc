//
// Copyright (C) 2006-2011 Christoph Sommer <christoph.sommer@uibk.ac.at>
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

#include "veins/modules/application/traci/TraCIDemo11p.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace veins;

Define_Module(veins::TraCIDemo11p);

void TraCIDemo11p::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        sentMessage = false;
        lastDroveAt = simTime();
        currentSubscribedServiceId = -1;
    }
    //std::cout << mobility->getExternalId() << std::endl;

    /*Py_Initialize();
    wchar_t** _argv = (wchar_t**) PyMem_Malloc(0);
    PySys_SetArgv(0, _argv);
    PyObject* myModule = PyImport_Import(PyUnicode_FromString("my-test"));
    PyObject* myFunction = PyObject_GetAttrString(myModule, "func");
    PyObject* args = PyTuple_Pack(1, PyLong_FromLong(2));
    PyObject* myResult = PyObject_CallObject(myFunction, args);
    long result = PyLong_AsLong(myResult);
    std::cout << "My Result: " << result << std::endl;
    Py_Finalize();*/

    /*py::scoped_interpreter guard{};
    py::module_ myTest = py::module_::import("my-test");
    py::object result_py = myTest.attr("func")(2);
    int result = result_py.cast<int>();
    std::cout << result << std::endl;*/

    /*py::scoped_interpreter guard{};
    py::module_ myTest = py::module_::import("my-test");
    py::str result_py = myTest.attr("test1")();
    std::string result = result_py;
    myTest.attr("test2")(result.c_str());*/
}

void TraCIDemo11p::onWSM(BaseFrame1609_4* frame)
{
    TraCIDemo11pMessage* wsm = check_and_cast<TraCIDemo11pMessage*>(frame);

    findHost()->getDisplayString().setTagArg("i", 1, "green");

    py::scoped_interpreter guard{};
    py::module_ myTest = py::module_::import("my-test");
    myTest.attr("test2")(wsm->getWeights());

    if (!sentMessage) {
        sentMessage = true;
        // repeat the received traffic update once in 2 seconds plus some random delay
        wsm->setSenderAddress(myId);
        wsm->setSerial(3);
        scheduleAt(simTime() + 2 + uniform(0.01, 0.2), wsm->dup());
    }
}

void TraCIDemo11p::handleSelfMsg(cMessage* msg)
{
    DemoBaseApplLayer::handleSelfMsg(msg);
}

void TraCIDemo11p::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);

    // stopped for for at least 10s?
    if (mobility->getSpeed() < 1) {
        if (simTime() - lastDroveAt >= 10 && sentMessage == false) {
            findHost()->getDisplayString().setTagArg("i", 1, "red");
            sentMessage = true;

            py::scoped_interpreter guard{};
            py::module_ myTest = py::module_::import("my-test");
            py::str result_py = myTest.attr("test1")();
            std::string result = result_py;

            TraCIDemo11pMessage* wsm = new TraCIDemo11pMessage();
            wsm->setWeights(result.c_str());
            populateWSM(wsm);

            // send right away on CCH, because channel switching is disabled
            sendDelayedDown(wsm, simTime() + 2 + uniform(0.01, 0.2));
        }
    } else {
        lastDroveAt = simTime();
    }
}
