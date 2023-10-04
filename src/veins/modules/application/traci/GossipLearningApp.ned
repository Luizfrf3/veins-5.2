
package org.car2x.veins.modules.application.traci;
import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

simple GossipLearningApp extends DemoBaseApplLayer
{
    @class(veins::GossipLearningApp);
    string appName = default("Gossip Learning App!");
}
