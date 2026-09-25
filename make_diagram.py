"""Generate DQN cycle diagram for LunarLander as PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

BLUE = '#1F3864'; RED = '#C00000'; TEAL = '#2E7D6C'; ORANGE = '#E76F00'
GRAY = '#555555'; LIGHTGRAY = '#F0F0F0'


def box(x, y, w, h, title, sub='', color=BLUE, subcolor=None, alpha=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 linewidth=2, edgecolor=color,
                                 facecolor=color, alpha=alpha))
    ax.text(x + w/2, y + h - 0.28, title,
            ha='center', va='top', fontsize=11, fontweight='bold', color=color)
    if sub:
        ax.text(x + w/2, y + 0.25, sub,
                ha='center', va='bottom', fontsize=8.5,
                color=subcolor or GRAY, style='italic')


def arrow(x1, y1, x2, y2, label='', color=GRAY, style='-|>', label_offset=(0, 0.15),
          curve='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle=style, mutation_scale=15,
                                  color=color, linewidth=1.4,
                                  connectionstyle=curve))
    if label:
        mx = (x1 + x2)/2 + label_offset[0]
        my = (y1 + y2)/2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=8.5,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.92))


ax.text(7, 8.7, 'Ciclo de entrenamiento — DQN sobre LunarLander-v3',
        ha='center', va='center', fontsize=14, fontweight='bold', color=BLUE)
ax.text(7, 8.3, 'MLP 8→128→128→4 · Replay 100k · Target sync 500 · ε: 1.0→0.02 en 100k pasos',
        ha='center', va='center', fontsize=9, color=GRAY, style='italic')

# --- COL IZQUIERDA (recolección) ---
ax.text(2.5, 7.85, 'COLECCIÓN DE DATOS',
        ha='center', fontsize=9.5, fontweight='bold', color=TEAL,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHTGRAY, edgecolor=TEAL))

box(0.5, 6.3, 4.0, 1.2, 'ENTORNO',
    'LunarLander-v3 · Box2D\nestado ∈ ℝ⁸  ·  4 acciones discretas', color=BLUE)

box(0.5, 4.5, 4.0, 1.5, 'POLÍTICA ε-GREEDY',
    'ε: 1.0 → 0.02 (decay lineal 100k pasos)\ncon prob ε: acción aleatoria\ncon prob 1-ε: argmax Q_online(s, ·)',
    color=BLUE)

box(0.5, 2.7, 4.0, 1.5, 'env.step(a)',
    '→ (s\', r, terminated, truncated)\nrecompensa densa: pad, patas,\ncombustible, aterrizaje/crash',
    color=BLUE)

# --- COL DERECHA (aprendizaje) ---
ax.text(11.5, 7.85, 'APRENDIZAJE',
        ha='center', fontsize=9.5, fontweight='bold', color=RED,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHTGRAY, edgecolor=RED))

box(9.5, 6.3, 4.0, 1.2, 'Q_target (congelada)',
    'copia de Q_online\ncada 500 pasos', color=RED)

box(9.5, 4.5, 4.0, 1.5, 'Q_online (MLP)',
    'Linear(8→128) → ReLU\nLinear(128→128) → ReLU\nLinear(128→4)\nAdam · lr=5e-4 · Huber',
    color=RED)

box(9.5, 2.7, 4.0, 1.5, 'BLANCO DE BELLMAN',
    'current_q = Q_online(s)[a]\ntarget_q = r + γ·max Q_target(s\', ·)·(1-term)\nloss = smooth_L1(current_q, target_q)',
    color=RED, subcolor=RED)

# --- CENTRO (replay) ---
box(5.5, 3.6, 3.0, 1.5, 'REPLAY BUFFER',
    'capacidad 100 000 · float32\n(s, a, r, s\', term)\nsample batch=64 cada paso',
    color=ORANGE)

# --- FLECHAS ---
arrow(2.5, 6.3, 2.5, 6.0, 'obs ∈ ℝ⁸', color=BLUE, label_offset=(0.85, 0))
arrow(2.5, 4.5, 2.5, 4.2, 'a', color=BLUE, label_offset=(0.25, 0))

# Loop de vuelta: env.step → entorno (drawn well outside the boxes)
arrow(0.5, 3.45, 0.15, 3.45, '', color=BLUE)
arrow(0.15, 3.45, 0.15, 7.1, '', color=BLUE)
arrow(0.15, 7.1, 0.5, 7.1, '', color=BLUE)
ax.text(0.32, 5.3, 's ← s\' ', fontsize=8, color=BLUE, fontweight='bold',
        rotation=90, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.95))

# Flecha step → replay buffer (transición)
arrow(4.5, 3.45, 5.5, 3.9, '', color=ORANGE)
ax.text(5.05, 3.35, '(s, a, r, s\', term)', fontsize=8, color=ORANGE,
        fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.95))

# Q_online → policy (leer valores para argmax)
arrow(9.5, 5.7, 4.5, 5.7, 'Q_online(s, ·)', color=RED,
      label_offset=(0.3, 0.25), curve='arc3,rad=-0.2')

# Q_online → Q_target (copia periódica, punteado)
line = Line2D([11.5, 11.5], [5.9, 6.4], linewidth=1.5, linestyle='--',
              color=RED, alpha=0.75)
ax.add_line(line)
arrow(11.5, 6.35, 11.5, 6.28, '', color=RED)
ax.text(12.85, 6.15, 'copia\ncada 500\nsteps', fontsize=7.5, color=RED,
        style='italic', ha='left', va='center')

# Replay → Blanco de Bellman (sample)
arrow(8.5, 4.15, 9.5, 3.55, 'sample (batch=64)', color=ORANGE,
      label_offset=(-0.05, 0.25))

# Q_online → Bellman (current_q)
arrow(10.4, 4.5, 10.4, 4.25, 'current_q', color=RED, label_offset=(-0.75, 0))

# Q_target → Bellman (next_q)
arrow(12.6, 6.3, 12.6, 4.25, 'next_q (max)', color=RED,
      label_offset=(0.8, 0), curve='arc3,rad=0.15')

# Bellman → Q_online (gradient)
arrow(11.0, 4.2, 11.0, 4.5, '∇L, Adam', color=RED, label_offset=(-0.8, 0),
      curve='arc3,rad=-0.15')

# Notes
ax.text(7, 1.5,
        'Nota: solo `terminated` (aterrizaje suave o crash) colapsa el bootstrap.\n'
        '`truncated` (corte a 1000 pasos) se guarda como done=0.',
        ha='center', fontsize=8, color=GRAY, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                  edgecolor='#F0AD00', linewidth=1))

ax.text(7, 0.3,
        'Ivan Enrique Rangel Santos — Taller 2 — Simulación y Aprendizaje por Refuerzo · Maestría en IA · Universidad de La Sabana',
        ha='center', fontsize=7.5, color=GRAY)

plt.tight_layout()
plt.savefig('/home/claude/lunarlander_final/figures/dqn_cycle_diagram.png', dpi=160,
            bbox_inches='tight', facecolor='white')
print('Diagram saved.')
