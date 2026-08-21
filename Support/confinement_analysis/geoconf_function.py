import matplotlib.pyplot as plt
import numpy as np

def peak_function(x, mu=0.3, beta=2.0):
    x = np.asarray(x)
    return (
        (x/mu)**mu * ((1 - x)/(1 - mu))**(1 - mu)
    ) ** beta

def loss_function(x=None, mu=0.3, beta=2.0):
    if x is None:
        x = np.linspace(0, 1.0, 1001)
    x = np.asarray(x)
    return 1 - peak_function(x, mu, beta)

# Example
x = np.linspace(0, 1.0, 1001)
beta = 2.0
y_05 = 1 - np.sqrt(4 * x * (1-x)) ** (beta)
y_10 = 1 - (x) ** (beta)

plt.figure()
print("┌──────────┬──────────┬──────────┐")
print("│    mu    │  peak_y  │  peak_x  │")
print("├──────────┼──────────┼──────────┤")
for mu in [0.1, 0.3, 0.5, 0.8, 1.0]:
    y = loss_function(x, mu, beta)
    print(f"│   {mu:.2f}   │   {np.max(y):.2f}   │   {np.argmax(y)/1000:.2f}   │")
    plt.plot(x, y, label=f"mu={mu}")
print("└──────────┴──────────┴──────────┘")
plt.plot(x, y_05, label="mu=0.5", linestyle="--", color="black")
plt.plot(x, y_10, label="mu=1.0", linestyle=":", color="black")
plt.ylim(0, 1.0)
plt.xlim(0, 1.0)
plt.xlabel(r"$\rho_{i,j}$")
plt.ylabel(r"$L_{surf}$")
plt.legend()
plt.show()