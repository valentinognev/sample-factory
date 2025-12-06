import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Try loading flight_log.csv from logs/ or current directory."""
    candidates = [
        "logs/RunnerDyn/data_log_20250714_140959.csv",
        "./logs/RunnerDyn/data_log_20250714_140959.csv",
        "data_log_20250714_140959.csv",
    ]
    for fname in candidates:
        if os.path.isfile(fname):
            return np.genfromtxt(fname, delimiter=",", names=True)
    print("Error: data_log_20250714_140959.csv not found in any of:", candidates)
    sys.exit(1)

def main():
    # Load recorded flight data
    data = load_data()
    A = 1 # amplification amplitude
    t = data["time"]
    agent_N = data["agent_N"]
    agent_E = data["agent_E"]
    target_N = data["target_N"]
    target_E = data["target_E"]
    r = data["r"]
    theta = data["theta"]
    radial_rate = data["radial_rate"]
    v_f_cmd = data["v_f_cmd"]
    v_r_cmd = data["v_r_cmd"]
    w_cmd = data["w_cmd"]
    vf = data["vf"]
    vr = data["vr"]
    w = data["w"]
    dN = data["dN"]
    dE = data["dE"]
    reward = data["reward"]

    # signal processing
    # E_a_n = np.average(rl_a_n)
    # E_a_e = np.average(rl_a_e)
    # rl_a_n = A * (rl_a_n - E_a_n)
    # rl_a_e = A * (rl_a_e - E_a_e)

    # 1. North component time-domain plot
    plt.figure()
    plt.plot(t, vf, label="v_f")
    plt.plot(t, vr,  label="v_r")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocities")
    plt.title("Vf/Vr")
    plt.legend()

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()
