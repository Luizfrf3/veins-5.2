
import org.car2x.veins.nodes.RSU;
import org.car2x.veins.nodes.Scenario;

simple CentralServer
{
    parameters:
        @display("i=block/routing");
    gates:
        inout gate[3];
}

network VehicleRSUServerScenario extends Scenario
{
    types:
        channel myChannel extends ned.DatarateChannel {
            delay = 10ms;
            datarate = 200Mbps;
        };

    submodules:
        rsu[3]: RSU {
            @display("p=150,140;i=veins/sign/yellowdiamond;is=vs");
        };
        server: CentralServer;
    connections allowunconnected:
        rsu[0].out --> myChannel --> server.gate$i[0];
        server.gate$o[0] --> myChannel --> rsu[0].in;
        rsu[1].out --> myChannel --> server.gate$i[1];
        server.gate$o[1] --> myChannel --> rsu[1].in;
        rsu[2].out --> myChannel --> server.gate$i[2];
        server.gate$o[2] --> myChannel --> rsu[2].in;
}
