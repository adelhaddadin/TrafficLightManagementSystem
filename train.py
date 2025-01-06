from simulation import TrafficLightDRLSimulationPrioritizedNoRepick
def main():
    cfg_path = r"C:\Program Files (x86)\Eclipse\Sumo\bin\simulation_test\city_network.sumocfg"
    sim = TrafficLightDRLSimulationPrioritizedNoRepick(cfg_path)
    sim.train()

if __name__ == "__main__":
    main()
