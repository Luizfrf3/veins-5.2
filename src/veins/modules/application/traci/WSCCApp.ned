
package org.car2x.veins.modules.application.traci;
import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

simple WSCCApp extends DemoBaseApplLayer
{
    @class(veins::WSCCApp);
    string appName = default("WSCC App!");
}
