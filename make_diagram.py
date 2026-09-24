"""Generate DQN cycle diagram as PNG."""
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


# Title
ax.text(7, 8.7, 'Ciclo de entrenamiento — DQN sobre Pong (Atari)',
        ha='center', va='center', fontsize=14, fontweight='bold', color=BLUE)
ax.text(7, 8.3, 'ALE/Pong-v5 · Nature CNN · Frame stack 4 · Replay 100k · Target sync 1000',
        ha='center', va='center', fontsize=9, color=GRAY, style='italic')

# --- COL IZQUIERDA (recolección) ---
ax.text(2.5, 7.85, 'COLECCIÓN DE DATOS',
        ha='center', fontsize=9.5, fontweight='bold', color=TEAL,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHTGRAY, edgecolor=TEAL))

box(0.5, 6.3, 4.0, 1.2, 'ENTORNO',
    'ALE/Pong-v5 · frame 210×160×3 RGB', color=BLUE)

box(0.5, 4.5, 4.0, 1.5, 'PREPROCESAMIENTO',
    'Noop reset 0-30 · Frame skip 4 (max sobre 2)\ngrayscale · resize 84×84 · frame stack 4\n→ (4, 84, 84) uint8',
    color=BLUE)

box(0.5, 2.7, 4.0, 1.5, 'POLÍTICA ε-GREEDY',
    'ε: 1.0 → 0.05 (decay lineal 250k pasos)\ncon prob ε: acción aleatoria\ncon prob 1-ε: argmax Q_online(s, ·)',
    color=BLUE)

box(0.5, 0.9, 4.0, 1.5, 'env.step(a)',
    '→ (s\', r ∈ {-1,0,+1}, terminated)\nrecompensa = +1 anota / -1 recibe',
    color=BLUE)

# --- COL DERECHA (aprendizaje) ---
ax.text(11.5, 7.85, 'APRENDIZAJE',
        ha='center', fontsize=9.5, fontweight='bold', color=RED,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHTGRAY, edgecolor=RED))

box(9.5, 6.3, 4.0, 1.2, 'Q_target (congelada)',
    'copia de Q_online\ncada 1 000 pasos', color=RED)

box(9.5, 4.5, 4.0, 1.5, 'Q_online (Nature CNN)',
    'Conv(32,k8,s4) → Conv(64,k4,s2) → Conv(64,k3,s1)\n→ Flatten(3136) → FC(512) → FC(6)\nAdam · lr=2.5e-4 · Huber loss',
    color=RED)

box(9.5, 2.7, 4.0, 1.5, 'BLANCO DE BELLMAN',
    'current_q = Q_online(s)[a]\ntarget_q = r + γ·max Q_target(s\', ·)·(1-term)\nloss = smooth_L1(current_q, target_q)',
    color=RED, subcolor=RED)

# --- CENTRO (replay) ---
box(5.5, 3.6, 3.0, 1.5, 'REPLAY BUFFER',
    'capacidad 100 000 · uint8\n(s, a, r, s\', term)\nsample batch=32 c/4 pasos',
    color=ORANGE)

# --- FLECHAS ---
# Col izquierda: entorno → prep → policy → step (bajando)
arrow(2.5, 6.3, 2.5, 6.0, 'obs cruda', color=BLUE, label_offset=(0.85, 0))
arrow(2.5, 4.5, 2.5, 4.2, 's = (4,84,84)', color=BLUE, label_offset=(1.1, 0))
arrow(2.5, 2.7, 2.5, 2.4, 'a', color=BLUE, label_offset=(0.25, 0))

# Loop de vuelta: env.step → PREPROC (siguiente frame)
arrow(0.5, 1.65, 0.15, 1.65, '', color=BLUE)
arrow(0.15, 1.65, 0.15, 5.25, '', color=BLUE)
arrow(0.15, 5.25, 0.5, 5.25, 's ← s\' (siguiente paso)', color=BLUE,
      label_offset=(1.15, 0.15))

# Flecha step → replay buffer (transición)
arrow(4.5, 1.65, 5.5, 3.9, '(s, a, r, s\', term)', color=ORANGE,
      label_offset=(0.1, 0.15))

# Q_online → policy (leer valores para argmax)
arrow(9.5, 5.1, 4.5, 3.35, 'Q_online(s, ·)', color=RED,
      label_offset=(0.3, 0.25), curve='arc3,rad=-0.15')

# Q_online → Q_target (copia periódica, punteado)
line = Line2D([11.5, 11.5], [5.9, 6.4], linewidth=1.5, linestyle='--',
              color=RED, alpha=0.75)
ax.add_line(line)
arrow(11.5, 6.35, 11.5, 6.28, '', color=RED)
ax.text(12.85, 6.15, 'copia\ncada 1000\nsteps', fontsize=7.5, color=RED,
        style='italic', ha='left', va='center')

# Replay → Blanco de Bellman (sample)
arrow(8.5, 4.15, 9.5, 3.55, 'sample (batch=32)', color=ORANGE,
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
        'Nota: solo `terminated` (real) colapsa el bootstrap.\n'
        '`truncated` (corte por tiempo) se guarda como done=0.',
        ha='center', fontsize=8, color=GRAY, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                  edgecolor='#F0AD00', linewidth=1))

ax.text(7, 0.3,
        'Ivan Enrique Rangel Santos — Taller 2 — Simulación y Aprendizaje por Refuerzo · Maestría en IA · Universidad de La Sabana',
        ha='center', fontsize=7.5, color=GRAY)

plt.tight_layout()
plt.savefig('/home/claude/pong_dqn/figures/dqn_cycle_diagram.png', dpi=160,
            bbox_inches='tight', facecolor='white')
print('Diagram saved.')
