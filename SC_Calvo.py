import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# --- MODEL PARAMETERS ---
T = 400
epsilon = 6.0
beta = 0.99
delta = 1.0

p_cs = 0.05  # Calm -> Storm
p_sc = 0.10  # Storm -> Calm

lambda_calm = 2.0
lambda_storm = 0.05
mc_base = 1.0
mc_shock = 2.0

rho_mc = 0.90  # Tedarik zinciri aksamasının kalıcılığı (Persistence)

# --- DATA GENERATION ---
np.random.seed(42)
weather = np.zeros(T)
lambdas = np.zeros(T)
thetas = np.zeros(T)
MCs = np.zeros(T)
inflation = np.zeros(T)
price_level = np.ones(T)
optimal_price = np.ones(T)

current_weather = 0
disruption = 0.0

for t in range(T):
    # 1. Weather Transitions
    if current_weather == 0:
        if np.random.rand() < p_cs:
            current_weather = 1
    else:
        if np.random.rand() < p_sc:
            current_weather = 0

    weather[t] = current_weather

    # 2. Logistics & Cost Shocks
    if current_weather == 1:
        lambdas[t] = lambda_storm
        disruption = 1.0
    else:
        lambdas[t] = lambda_calm
        disruption = disruption * rho_mc

    MCs[t] = mc_base + (mc_shock * disruption)

    # 3. Calvo Parameter
    thetas[t] = np.exp(-lambdas[t] * delta)

    # 4. Pricing Dynamics
    if t > 0:
        markup = epsilon / (epsilon - 1)
        optimal_price[t] = MCs[t] * markup

        p_term1 = thetas[t] * (price_level[t - 1] ** (1 - epsilon))
        p_term2 = (1 - thetas[t]) * (optimal_price[t] ** (1 - epsilon))
        price_level[t] = (p_term1 + p_term2) ** (1 / (1 - epsilon))

        inflation[t] = ((price_level[t] - price_level[t - 1]) / price_level[t - 1]) * 100
    else:
        price_level[t] = mc_base * (epsilon / (epsilon - 1))

# --- VISUALIZATION & ANIMATION ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 14))  # Biraz daha büyüttük ki 4 grafik rahat sığsın
fig.canvas.manager.set_window_title('Shipping-Calendar Calvo Model')
fig.patch.set_facecolor('#FAFAFA')

# Grid Layout: 3 Satır (1 Harita + 2 Satır Grafik)
gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.2, 1.2], hspace=0.35, wspace=0.15)
ax_map = fig.add_subplot(gs[0, :])
ax_theta = fig.add_subplot(gs[1, 0])
ax_mc = fig.add_subplot(gs[1, 1])
ax_price = fig.add_subplot(gs[2, 0])  # YENİ: Fiyat Seviyesi Grafiği
ax_inf = fig.add_subplot(gs[2, 1])  # Enflasyon Grafiği (Sağ Alta Alındı)

# --- 1. THE ISLAND & SEA (Top Map) ---
ax_map.set_xlim(0, 10)
ax_map.set_ylim(0, 5)
ax_map.axis('off')

beach = patches.Ellipse((8, 2.5), 4.0, 5.0, color='#F3E5AB', zorder=1)
ax_map.add_patch(beach)
island = patches.Ellipse((8, 2.5), 3.2, 4.2, color='#66BB6A', ec='#388E3C', lw=2, zorder=2)
ax_map.add_patch(island)
ax_map.text(8, 2.5, "ISLAND\nECONOMY", color="white", fontsize=15, ha='center', va='center', zorder=3,
            fontweight='bold')

port = patches.Rectangle((5.8, 2.1), 0.8, 0.8, color='#8D6E63', ec='#5D4037', lw=2, zorder=2)
ax_map.add_patch(port)
ax_map.text(5.5, 1.8, "Port", color="black", fontsize=11, fontweight='bold', zorder=3)

ships = []
ship_scatter = ax_map.scatter([], [], marker='s', s=250, color='#E53935', ec='white', lw=2, zorder=4)

weather_text = ax_map.text(0.5, 4.2, "", fontsize=18, fontweight='bold', ha='left',
                           bbox=dict(facecolor='#FFFFFF', alpha=0.9, edgecolor='#B0BEC5', boxstyle='round,pad=0.5',
                                     lw=2))

# --- 2. THE GRAPHS ---
x_data = np.arange(T)

# Theta Plot
line_theta, = ax_theta.plot([], [], color='#8E24AA', lw=3)
ax_theta.set_xlim(0, 100)
ax_theta.set_ylim(-0.1, 1.2)
ax_theta.set_title(r"Price Stickiness (Calvo $\theta_t$)", fontsize=14, fontweight='bold', color='#333333')
ax_theta.fill_between(x_data, thetas, color='#8E24AA', alpha=0.15)
ax_theta.text(0.03, 0.08, r'$\theta_t = e^{-\lambda_t \Delta}$', transform=ax_theta.transAxes,
              fontsize=16,
              bbox=dict(facecolor='#FFFFFF', alpha=0.95, edgecolor='#8E24AA', boxstyle='round,pad=0.4', lw=2),
              zorder=10)

# Marginal Cost Plot
line_mc, = ax_mc.plot([], [], color='#E53935', lw=3)
ax_mc.set_xlim(0, 100)
ax_mc.set_ylim(0.5, 4.0)
ax_mc.set_title(r"Marginal Cost ($MC_t$)", fontsize=14, fontweight='bold', color='#333333')
ax_mc.fill_between(x_data, MCs, mc_base, color='#E53935', alpha=0.15)
ax_mc.text(0.03, 0.75, r'$MC_t \propto \frac{(S_t P_t^*)^\alpha}{A_t}$', transform=ax_mc.transAxes,
           fontsize=16, bbox=dict(facecolor='#FFFFFF', alpha=0.95, edgecolor='#E53935', boxstyle='round,pad=0.4', lw=2),
           zorder=10)

# YENİ: Price Level Plot
line_price, = ax_price.plot([], [], color='#00897B', lw=3)  # Teal rengi
ax_price.set_xlim(0, 100)
ax_price.set_ylim(1.0, max(price_level) + 0.5)
ax_price.set_title(r"Aggregate Price Level ($P_t$)", fontsize=14, fontweight='bold', color='#333333')
ax_price.fill_between(x_data, price_level, min(price_level), color='#00897B', alpha=0.15)
ax_price.text(0.03, 0.75,
              r'$P_t = [ \theta_t P_{t-1}^{1-\epsilon} + (1-\theta_t)(P_t^\#)^{1-\epsilon} ]^{\frac{1}{1-\epsilon}}$',
              transform=ax_price.transAxes, fontsize=14,
              bbox=dict(facecolor='#FFFFFF', alpha=0.95, edgecolor='#00897B', boxstyle='round,pad=0.4', lw=2),
              zorder=10)

# Inflation Plot
line_inf, = ax_inf.plot([], [], color='#FB8C00', lw=3)
ax_inf.set_xlim(0, 100)
ax_inf.set_ylim(min(inflation) - 2, max(inflation) + 5)
ax_inf.set_title(r"Inflation ($\pi_t$) - The Repricing Wave", fontsize=14, fontweight='bold', color='#333333')
ax_inf.axhline(0, color='black', lw=1.5, ls='--')
ax_inf.fill_between(x_data, inflation, 0, where=(inflation >= 0), color='#FB8C00', alpha=0.25)
ax_inf.fill_between(x_data, inflation, 0, where=(inflation < 0), color='#1976D2', alpha=0.25)
ax_inf.text(0.03, 0.75, r'$\pi_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100$',
            transform=ax_inf.transAxes, fontsize=16,
            bbox=dict(facecolor='#FFFFFF', alpha=0.95, edgecolor='#FB8C00', boxstyle='round,pad=0.4', lw=2), zorder=10)

for ax in [ax_theta, ax_mc, ax_price, ax_inf]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_facecolor('#FFFFFF')


# --- 3. ANIMATION LOGIC ---
def update(frame):
    if frame > 100:
        ax_theta.set_xlim(frame - 100, frame)
        ax_mc.set_xlim(frame - 100, frame)
        ax_price.set_xlim(frame - 100, frame)
        ax_inf.set_xlim(frame - 100, frame)

    line_theta.set_data(x_data[:frame], thetas[:frame])
    line_mc.set_data(x_data[:frame], MCs[:frame])
    line_price.set_data(x_data[:frame], price_level[:frame])
    line_inf.set_data(x_data[:frame], inflation[:frame])

    global ships

    if weather[frame] == 1:
        ax_map.set_facecolor('#CFD8DC')
        weather_text.set_text("STATUS: STORM\n(Port Closed, $\\lambda_t \\downarrow$)")
        weather_text.set_color("#C62828")
    else:
        ax_map.set_facecolor('#E1F5FE')
        weather_text.set_text("STATUS: CALM\n(Supply Flowing, $\\lambda_t \\uparrow$)")
        weather_text.set_color("#1565C0")

        if np.random.rand() < 0.35:
            ships.append([0.0, 2.5 + (np.random.rand() * 1.5 - 0.75)])

    active_ships = []
    for s in ships:
        if weather[frame] == 0:
            s[0] += 0.25

        if s[0] < 5.8:
            active_ships.append(s)

    ships = active_ships

    if len(ships) > 0:
        ship_scatter.set_offsets(ships)
    else:
        ship_scatter.set_offsets(np.empty((0, 2)))

    return line_theta, line_mc, line_price, line_inf, ship_scatter, weather_text


ani = animation.FuncAnimation(fig, update, frames=T, interval=120, blit=False)

plt.tight_layout()
plt.show()