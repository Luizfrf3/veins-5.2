
import org.car2x.veins.nodes.RSU;
import org.car2x.veins.nodes.Scenario;

simple HybridMethodServer
{
    parameters:
        @display("i=block/routing");
    gates:
        inout gate[22];
}

network HybridMethodScenario extends Scenario
{
    types:
        channel myChannel extends ned.DatarateChannel {
            delay = 100ms;
            datarate = 500Mbps;
        };

    submodules:
        //rsu[7]: RSU {
        //rsu[8]: RSU {
        //rsu[12]: RSU {
        rsu[22]: RSU {
            @display("p=150,140;i=veins/sign/yellowdiamond;is=vs;r=500");
        };
        server: HybridMethodServer;
    connections allowunconnected:
        rsu[0].out --> myChannel --> server.gate$i[0];
        server.gate$o[0] --> myChannel --> rsu[0].in;
        rsu[1].out --> myChannel --> server.gate$i[1];
        server.gate$o[1] --> myChannel --> rsu[1].in;
        rsu[2].out --> myChannel --> server.gate$i[2];
        server.gate$o[2] --> myChannel --> rsu[2].in;
        rsu[3].out --> myChannel --> server.gate$i[3];
        server.gate$o[3] --> myChannel --> rsu[3].in;
        rsu[4].out --> myChannel --> server.gate$i[4];
        server.gate$o[4] --> myChannel --> rsu[4].in;
        rsu[5].out --> myChannel --> server.gate$i[5];
        server.gate$o[5] --> myChannel --> rsu[5].in;
        rsu[6].out --> myChannel --> server.gate$i[6];
        server.gate$o[6] --> myChannel --> rsu[6].in;
        // For 7 RSUs, remove the ones below
        rsu[7].out --> myChannel --> server.gate$i[7];
        server.gate$o[7] --> myChannel --> rsu[7].in;
        // For 8 RSUs, remove the ones below
        rsu[8].out --> myChannel --> server.gate$i[8];
        server.gate$o[8] --> myChannel --> rsu[8].in;
        rsu[9].out --> myChannel --> server.gate$i[9];
        server.gate$o[9] --> myChannel --> rsu[9].in;
        rsu[10].out --> myChannel --> server.gate$i[10];
        server.gate$o[10] --> myChannel --> rsu[10].in;
        rsu[11].out --> myChannel --> server.gate$i[11];
        server.gate$o[11] --> myChannel --> rsu[11].in;
        // For 12 RSUs, remove the ones below
        rsu[12].out --> myChannel --> server.gate$i[12];
        server.gate$o[12] --> myChannel --> rsu[12].in;
        rsu[13].out --> myChannel --> server.gate$i[13];
        server.gate$o[13] --> myChannel --> rsu[13].in;
        rsu[14].out --> myChannel --> server.gate$i[14];
        server.gate$o[14] --> myChannel --> rsu[14].in;
        rsu[15].out --> myChannel --> server.gate$i[15];
        server.gate$o[15] --> myChannel --> rsu[15].in;
        rsu[16].out --> myChannel --> server.gate$i[16];
        server.gate$o[16] --> myChannel --> rsu[16].in;
        rsu[17].out --> myChannel --> server.gate$i[17];
        server.gate$o[17] --> myChannel --> rsu[17].in;
        rsu[18].out --> myChannel --> server.gate$i[18];
        server.gate$o[18] --> myChannel --> rsu[18].in;
        rsu[19].out --> myChannel --> server.gate$i[19];
        server.gate$o[19] --> myChannel --> rsu[19].in;
        rsu[20].out --> myChannel --> server.gate$i[20];
        server.gate$o[20] --> myChannel --> rsu[20].in;
        rsu[21].out --> myChannel --> server.gate$i[21];
        server.gate$o[21] --> myChannel --> rsu[21].in;
}
