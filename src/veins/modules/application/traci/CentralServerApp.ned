
package org.car2x.veins.modules.application.traci;
import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

simple CentralServerApp extends DemoBaseApplLayer
{
    @class(veins::CentralServerApp);
    string appName = default("Central Server App!");
}
