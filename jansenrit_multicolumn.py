from pyrates import CircuitTemplate
import matplotlib.pyplot as plt

template = CircuitTemplate.from_yaml("model_definitions/ThreeColumnNetwork")
res = template.run(simulation_time=100.0, step_size=1e-3,
                   outputs={"v_e": "all/pc/rpo_e_in/v", "v_i": "all/pc/rpo_i/v"})
v = res["v_e"].values + res["v_i"].values

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res.index, v)
ax.legend(["C1", "C2", "C3"])
ax.set_xlabel("time")
ax.set_ylabel("V")
ax.set_title("3-Column Jansen-Rit Model Dynamics")
plt.show()
