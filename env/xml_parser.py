import xml.etree.ElementTree as ET

def parse_network(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    network_data = {
        'segments': [],
        'junctions': {},
        'connections': []
    }

    # Parse edges and lanes
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        lanes = []
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            speed = float(lane.get('speed'))
            length = float(lane.get('length'))
            lanes.append({'id': lane_id, 'speed': speed, 'length': length})
        network_data['segments'].append({'id': edge_id, 'lanes': lanes})

    # Parse junctions
    for junction in root.findall('junction'):
        junction_id = junction.get('id')
        int_lanes = junction.get('intLanes').split() if junction.get('intLanes') else []
        inc_lanes = junction.get('incLanes').split() if junction.get('incLanes') else []
        network_data['junctions'][junction_id] = {
            'int_lanes': int_lanes,
            'inc_lanes': inc_lanes
        }

    # Parse connections
    for connection in root.findall('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        from_lane = connection.get('fromLane')
        to_lane = connection.get('toLane')
        via = connection.get('via')
        network_data['connections'].append({
            'from': from_edge, 
            'to': to_edge, 
            'from_lane': from_lane, 
            'to_lane': to_lane, 
            'via': via
        })

    return network_data

network_data = parse_network('/Users/cheimamezdour/Projects/PFE/DQN-ITSCwPD/env/custom_env/data/J7_TLS/J7_TLS.net.xml')

