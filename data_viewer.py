import traci

traci.start(["sumo", "-c", "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\simulation_test\\city_network.sumocfg"])
tl_ids = list(traci.trafficlight.getIDList())
traci.close()

print("Traffic Light IDs:", tl_ids)

lane_ids = []
traci.start(["sumo", "-c", "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\simulation_test\\city_network.sumocfg"])
for tl_id in tl_ids:
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    lane_ids.extend(lanes)
traci.close()

print("Lane IDs:", lane_ids)

