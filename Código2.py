import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from numba import njit

# Constantes
G = 1.0
mT, mL = 8.0, 4.0
tempo = np.linspace(0, 60, 1500)
dt = tempo[1] - tempo[0]

# Condições iniciais
y0 = np.array([0, 0, 10, 0, 0, 0, 0, 0])

# Malha circular (pontos da maré)
n_pontos = 100
raio_base = 1.5
theta = np.linspace(0, 2*np.pi, n_pontos, endpoint=False)
x_rel = raio_base * np.cos(theta)
y_rel = raio_base * np.sin(theta)

# Função do sistema Terra-Lua
@njit
def sistema(y, t, G, m1, m2):
    xT, yT, xL, yL = y[0:4]
    vxT, vyT, vxL, vyL = y[4:8]

    dx, dy = xL - xT, yL - yT
    r = np.sqrt(dx**2 + dy**2) + 1e-8

    aTx = G * m2 * dx / r**3
    aTy = G * m2 * dy / r**3
    aLx = -G * m1 * dx / r**3
    aLy = -G * m1 * dy / r**3

    return np.array([vxT, vyT, vxL, vyL, aTx, aTy, aLx, aLy])

# Solução
solucao = odeint(sistema, y0, tempo, args=(G, mT, mL))
xT, yT = solucao[:, 0], solucao[:, 1]
xL, yL = solucao[:, 2], solucao[:, 3]

# Inicializar deformações da mare
x_def = np.zeros((len(tempo), n_pontos))
y_def = np.zeros((len(tempo), n_pontos))
x_def[0, :] = x_rel
y_def[0, :] = y_rel
x_def[1, :] = x_rel
y_def[1, :] = y_rel

# Integração de Verlet (força de maré)
for i in range(1, len(tempo)-1):
    xTi, yTi = xT[i], yT[i]
    xLi, yLi = xL[i], yL[i]

    for j in range(n_pontos):
        # Posição do ponto e da Terra
        xp = x_def[i, j] + xTi
        yp = y_def[i, j] + yTi

        dxp, dyp = xLi - xp, yLi - yp
        dxc, dyc = xLi - xTi, yLi - yTi

        rp = np.sqrt(dxp**2 + dyp**2) + 1e-8
        rc = np.sqrt(dxc**2 + dyc**2) + 1e-8

        # Força de maré
        ax = G * mL * (dxp / rp**3 - dxc / rc**3)
        ay = G * mL * (dyp / rp**3 - dyc / rc**3)

        # Verlet nos deslocamentos
        x_def[i+1, j] = 2*x_def[i, j] - x_def[i-1, j] + ax * dt**2
        y_def[i+1, j] = 2*y_def[i, j] - y_def[i-1, j] + ay * dt**2

# Animação
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)

terra = Circle((xT[0], yT[0]), radius=1, edgecolor='green', fill=False, linestyle='-')
lua = Circle((xL[0], yL[0]), radius=0.2, edgecolor='red', fill=False, linestyle='-')
maré, = ax.plot([], [], 'b-', markersize=1, zorder=1)

ax.add_patch(terra)
ax.add_patch(lua)

def animate(i):
    x_abs = x_def[i] + xT[i]
    y_abs = y_def[i] + yT[i]

    terra.center = (xT[i], yT[i])
    lua.center = (xL[i], yL[i])
    maré.set_data(x_abs, y_abs)

    Rx_cm = (xT[i] + xL[i]) / 2
    Ry_cm = (yT[i] + yL[i]) / 2
    ax.set_xlim(Rx_cm - 6, Rx_cm + 6)
    ax.set_ylim(Ry_cm - 6, Ry_cm + 6)

    return terra, lua, maré

ani = FuncAnimation(fig, animate, frames=len(tempo), interval=25, blit=True)
plt.show()
