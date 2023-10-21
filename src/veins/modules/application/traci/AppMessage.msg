
import veins.base.utils.Coord;
import veins.modules.messages.BaseFrame1609_4;
import veins.base.utils.SimpleAddress;

namespace veins;

packet AppMessage extends BaseFrame1609_4 {
	string weights;
	int datasetSize;
    LAddress::L2Type senderAddress = -1;
    string senderId;
    int dest;
    bool isRSU = false;
}
