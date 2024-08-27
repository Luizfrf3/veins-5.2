import json
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

tree = ET.parse('lust.net.xml')
root = tree.getroot()

net_edges = set()
for child in root:
    if child.tag == 'edge':
        net_edges.add(child.attrib['id'])

paths = []
departs = []
for file in ['local.0.rou.xml', 'local.1.rou.xml', 'local.2.rou.xml']:
    tree = ET.parse('../../../../../LuSTScenario/scenario/DUARoutes/' + file)
    root = tree.getroot()
    for vehicle in root:
        for route in vehicle:
            route_edges = route.attrib['edges'].split(' ')
            i = 0
            found = False
            path = []
            while i < len(route_edges):
                if found:
                    if route_edges[i] in net_edges:
                        path.append(route_edges[i])
                    else:
                        break
                else:
                    found = route_edges[i] in net_edges
                    if found:
                        path.append(route_edges[i])
                i += 1
            if found:
                departs.append(vehicle.attrib['depart'])
                paths.append(path)

tmp = []
for path in paths:
    tmp.append(len(path))
print(min(tmp))
print(max(tmp))
for i in range(40):
    print(i, tmp.count(i))

data = []
for i in range(len(paths)):
    if len(paths[i]) >= 10:
        data.append({
            'size': len(paths[i]),
            'depart': departs[i],
            'route': ' '.join(paths[i])
        })
with open('dua-routes.json', 'w') as f:
    json.dump(data, f, indent=4)

random.seed(123)
random.shuffle(data)
root = ET.Element('routes')
for i in range(100):
    vehicle = ET.SubElement(root, 'vehicle', id=('v' + str(i)), depart=(str(i * 2) + '.0'), departPost='random', departSpeed='max')
    ET.SubElement(vehicle, 'route', edges=data[i]['route'])
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
with open('routes-generated.xml', 'w') as f:
    f.write(xmlstr)
