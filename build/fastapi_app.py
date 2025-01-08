import sys
import threading
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traci
from shared_data import (
    shared_inference,
    shared_inference_lock,
    shared_metrics,
    shared_metrics_lock,
    shared_emergency,
    shared_emergency_lock,
)

##############################################################################
# Configuration
##############################################################################

# Avoid using 'dir' as it's a built-in function name
simulation_dir = r"C:\Users\Azd_A\OneDrive\Desktop\TrafficLightManagementSystem"
sys.path.append(simulation_dir)

# SUMO Configuration for Simulation 13 (no GUI)
sumo_cfg_13 = r"C:\Users\Azd_A\OneDrive\Desktop\SUMO\simulation13\city_network.sumocfg"
sumo_cmd_13 = ["sumo", "-c", sumo_cfg_13, "--step-length", "0.24"]

##############################################################################
# Data Structures for sim13
##############################################################################

# Data for simulation13
sim13_data = {
    "time": 0.0,
    "co2": 0.0,
    "nox": 0.0,
    "pmx": 0.0,
    "vehicle_count": 0,
    "average_speed": 0.0,
}

# Lock for sim13 data
sim13_lock = threading.Lock()

##############################################################################
# Threads
##############################################################################

def run_sim13():
    """
    Start and run the 'simulation13' config in headless (no GUI) mode
    under label='sim13'. We'll gather data similarly.
    """
    try:
        traci.start(sumo_cmd_13, label="sim13")
        conn_13 = traci.getConnection("sim13")
    except traci.exceptions.TraCIException as e:
        print(f"Error starting sim13: {e}")
        return

    try:
        while conn_13.simulation.getMinExpectedNumber() > 0:
            start_time = time.time()

            # Temporary accumulators for this interval
            interval_co2 = 0.0
            interval_nox = 0.0
            interval_pmx = 0.0
            interval_speed = 0.0
            interval_vehicle_count = 0

            # Collect data for ~3 seconds
            while time.time() - start_time < 3:
                conn_13.simulationStep()

                vehicle_ids = conn_13.vehicle.getIDList()
                current_vehicle_count = len(vehicle_ids)
                interval_vehicle_count = current_vehicle_count  # Capture current count

                for vehicle_id in vehicle_ids:
                    interval_co2 += conn_13.vehicle.getCO2Emission(vehicle_id)
                    interval_nox += conn_13.vehicle.getNOxEmission(vehicle_id)
                    interval_pmx += conn_13.vehicle.getPMxEmission(vehicle_id)
                    interval_speed += conn_13.vehicle.getSpeed(vehicle_id)

                time.sleep(0.1)  # Adjust as needed

            avg_speed = (
                interval_speed / interval_vehicle_count
                if interval_vehicle_count > 0
                else 0.0
            )

            # Update sim13_data with thread safety
            with sim13_lock:
                sim13_data["time"] = conn_13.simulation.getTime()
                sim13_data["co2"] += interval_co2
                sim13_data["nox"] += interval_nox
                sim13_data["pmx"] += interval_pmx
                # Update vehicle_count to current count instead of accumulating
                sim13_data["vehicle_count"] = interval_vehicle_count
                # Average speed is already set to the current interval's average
                sim13_data["average_speed"] = avg_speed

    except Exception as e:
        print(f"Exception in run_sim13: {e}")
    finally:
        print("Closing 'sim13' connection")
        conn_13.close()
        
##############################################################################
# Start sim13 in a Separate Thread
##############################################################################

sim13_thread = threading.Thread(target=run_sim13, daemon=True)

sim13_thread.start()

##############################################################################
# FastAPI Application
##############################################################################
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/emissions")
async def get_emissions():
    """
    Returns current emissions data for both simulations in one JSON response.
    - sim13: Calculated locally in this file.
    - shared_sim: Pulled from shared_data.py.
    """
    # Retrieve shared simulation emissions data
    with shared_metrics_lock:
        shared_emissions = {
            "time": shared_metrics.get("time", 0.0),
            "co2": shared_metrics.get("co2", 0.0),
            "nox": shared_metrics.get("nox", 0.0),
            "pmx": shared_metrics.get("pmx", 0.0),
        }

    # Retrieve sim13 emissions data
    with sim13_lock:
        sim13_emissions = {
            "time": sim13_data.get("time", 0.0),
            "co2": sim13_data.get("co2", 0.0),
            "nox": sim13_data.get("nox", 0.0),
            "pmx": sim13_data.get("pmx", 0.0),
        }

    return JSONResponse(content={
        "shared_sim": shared_emissions,
        "sim13": sim13_emissions,
    })


@app.get("/vehicles")
async def get_vehicle_data():
    """
    Returns current vehicle count and average speed for both simulations.
    - sim13: Calculated locally in this file.
    - shared_sim: Pulled from shared_data.py.
    """
    # Retrieve shared simulation vehicle data
    with shared_metrics_lock:
        shared_vehicle_data = {
            "vehicle_count": shared_metrics.get("vehicle_count", 0),
            "average_speed": shared_metrics.get("average_speed", 0.0),
        }

    # Retrieve sim13 vehicle data
    with sim13_lock:
        sim13_vehicle_data = {
            "vehicle_count": sim13_data.get("vehicle_count", 0),
            "average_speed": sim13_data.get("average_speed", 0.0),
        }

    return JSONResponse(content={
        "shared_sim": shared_vehicle_data,
        "sim13": sim13_vehicle_data,
    })


@app.get("/status")
async def get_status():
    """
    Check if the sim13 simulation's thread is running (alive) or has stopped.
    """
    status_13 = "running" if sim13_thread.is_alive() else "stopped"
    return JSONResponse(content={
        "sim13": status_13
    })


@app.get("/occupancy")
def get_occupancy():
    """
    Return the latest predicted occupancy (or any relevant inference)
    for each lane group: A, B, C, D from the shared data.
    """
    with shared_inference_lock:
        data = {
            "A": shared_inference.get("A", 0),
            "B": shared_inference.get("B", 0),
            "C": shared_inference.get("C", 0),
            "D": shared_inference.get("D", 0)
        }
    return JSONResponse(content=data)


@app.get("/emergency")
def get_emergency():
    """
    Return the latest emergency status for each lane group: A, B, C, D from the shared data.
    """
    with shared_emergency_lock:
        data = {
            "emerg_veh_id": shared_emergency.get("emerg_veh_id", ""),
            "emerg_lane_id": shared_emergency.get("emerg_lane_id", ""),
            "controlling_tl_id": shared_emergency.get("controlling_tl_id", "")
        }
    return JSONResponse(content=data)


# Optional: run via "python fastapi_app.py" for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
