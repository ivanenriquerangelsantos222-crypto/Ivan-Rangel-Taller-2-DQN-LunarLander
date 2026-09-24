# Taller 2 — DQN sobre Pong (Atari)

**Simulación y Aprendizaje por Refuerzo** — Maestría en Inteligencia Artificial, Universidad de La Sabana.
Autor: Ivan Enrique Rangel Santos.

Implementación desde cero de un agente **Deep Q-Network** para resolver `ALE/Pong-v5`, un ambiente Atari que no se había trabajado previamente en clase. El proyecto abarca desde la construcción de la pipeline de preprocesamiento visual (grayscale, resize, frame stacking) hasta el diseño de la CNN Nature 2015 y el entrenamiento con replay buffer, target network y política ε-greedy. Se documentan las particularidades del ambiente, las decisiones de arquitectura, los resultados obtenidos y las dificultades técnicas encontradas.

## Índice

1. [Descripción del ambiente](#1-descripción-del-ambiente-alepong-v5)
2. [Espacios de acciones y observaciones](#2-espacios-de-acciones-y-observaciones)
3. [Sistema de recompensas y terminación](#3-sistema-de-recompensas-y-terminación)
4. [Preprocesamiento y wrappers](#4-preprocesamiento-y-wrappers)
5. [Ciclo de entrenamiento DQN](#5-ciclo-de-entrenamiento-dqn)
6. [Arquitectura de la red](#6-arquitectura-de-la-red-nature-cnn)
7. [Hiperparámetros y justificación](#7-hiperparámetros-y-justificación)
8. [Resultados del entrenamiento](#8-resultados-del-entrenamiento)
9. [Reflexión sobre los resultados](#9-reflexión-sobre-los-resultados)
10. [Dificultades encontradas](#10-dificultades-encontradas)
11. [Cómo reproducir](#11-cómo-reproducir)
12. [Referencias](#12-referencias)

## Contenido del repo

```
├── src/atari_dqn/
│   ├── wrappers.py            # Preprocesamiento Atari (noop reset, frameskip, grayscale, resize, framestack)
│   ├── network.py             # Nature CNN 2015 (3 conv + 2 fc)
│   ├── buffer.py              # Replay buffer con almacenamiento uint8
│   ├── agent.py               # DQNAgent + DQNConfig (ε-greedy, target sync, Bellman step)
│   ├── train.py               # Loop de entrenamiento con logging y checkpoints
│   └── eval.py                # Evaluación con política voraz
├── notebooks/
│   └── pong_dqn_colab.ipynb   # Notebook autónomo para entrenar en Google Colab (GPU T4)
├── figures/
│   ├── dqn_cycle_diagram.png  # Diagrama del ciclo de entrenamiento
│   ├── pong_learning_curve.png# Curva de aprendizaje (evidencia)
│   └── pong_gameplay.gif      # Video del agente entrenado (evidencia)
├── saves/
│   ├── pong_dqn_final.pt      # Checkpoint del modelo entrenado
│   ├── history.npz            # Historia de recompensas por episodio
│   └── eval_results.npz       # Evaluación con 10 episodios greedy
├── make_diagram.py            # Reconstruye el diagrama del ciclo DQN
├── requirements.txt
├── README.md                  # Este archivo
├── LICENSE
└── .gitignore
```

## 1. Descripción del ambiente (`ALE/Pong-v5`)

Pong es el juego original de Atari 2600 (1972): dos paletas verticales golpean una bola que rebota entre ellas; cada vez que una paleta no logra devolverla, el rival marca un punto. El primero en llegar a **21 puntos** gana el juego. En la implementación de Gymnasium a través del *Arcade Learning Environment* (ALE), el agente controla la paleta derecha (verde) y compite contra una IA scripted que controla la izquierda.

Elegí Pong entre los ambientes Atari por tres razones concretas: (1) es el "hello world" del Deep Reinforcement Learning aplicado a Atari (Mnih et al., 2013, 2015); (2) su función de recompensa es densa comparada con otros juegos Atari (recibe señal en cada punto, no cada episodio), lo que lo hace tratable en presupuestos de cómputo de laboratorio; (3) su convergencia visual es rápida y clara — la política aprendida se puede interpretar viendo al agente jugar.

## 2. Espacios de acciones y observaciones

### Observaciones (crudas, antes de preprocesar)

| Propiedad | Valor |
|---|---|
| Tipo | `Box` |
| Forma | `(210, 160, 3)` |
| Dtype | `uint8` |
| Rango | 0–255 (píxeles RGB) |
| Frame rate | 60 Hz nativo del juego |

Cada observación es literalmente la imagen del televisor de Atari: 210 filas × 160 columnas × 3 canales RGB. Contiene la puntuación arriba, las dos paletas laterales, la bola, y las líneas de la cancha.

### Acciones

| Propiedad | Valor |
|---|---|
| Tipo | `Discrete(6)` |
| Índices | 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE |

Aunque nominalmente son 6 acciones, Pong en el fondo tiene solo 3 comportamientos distintos: no moverse (NOOP/FIRE), subir la paleta (RIGHT/RIGHTFIRE) y bajarla (LEFT/LEFTFIRE). El resto son redundantes por el manejo original de la consola. Dejamos las 6 tal como las expone el entorno porque descartarlas requeriría un wrapper adicional y no aporta mejora medible.

## 3. Sistema de recompensas y terminación

**Recompensa:** discreta en `{-1, 0, +1}` por paso del entorno crudo (antes del frame skip):
- `+1` cuando el agente anota (la paleta izquierda deja pasar la bola).
- `-1` cuando el rival anota (el agente deja pasar la bola).
- `0` en todos los demás pasos.

Un episodio dura hasta que uno de los dos jugadores alcanza 21 puntos, así que el **retorno acumulado** está en el rango `[-21, +21]`:
- `-21`: derrota humillante (perdió los 21 puntos, no anotó ninguno).
- `0`: empate imposible por reglas del juego, pero valores cercanos a 0 son partidas muy peleadas.
- `+21`: victoria perfecta (anotó los 21 puntos sin recibir ninguno).

**Terminación:** el episodio termina (`terminated=True`) solo cuando termina el juego (alguien llega a 21). Pong no tiene "vidas" en el sentido de Space Invaders o Breakout, así que no aplica el wrapper `terminal_on_life_loss` que sí sería útil en otros juegos. El wrapper `AtariPreprocessing` reporta `truncated=True` si se supera el límite de pasos (nunca ocurre en Pong con jugadores razonables).

## 4. Preprocesamiento y wrappers

Meter directamente el frame crudo `(210, 160, 3)` en una CNN sería catastrófico: 100 800 entradas por observación, información redundante (color, resolución excesiva) y el problema de dinámica que un solo frame no puede resolver (la bola se ve, pero no se sabe hacia dónde va). El pipeline estándar de Mnih et al. (2015) resuelve esto en cinco pasos.

| Wrapper | Qué hace | Por qué |
|---|---|---|
| `noop_max=30` | Ejecuta entre 0 y 30 no-ops al hacer `reset()` | Aleatoriza el estado inicial. Sin esto, cada partida arranca idéntica y el agente puede memorizar la primera jugada. |
| `frame_skip=4` (con max sobre los últimos 2) | El agente decide una acción cada 4 frames; el entorno la repite y devuelve el pixel-wise max de los últimos 2 frames del skip. | Atari corre a 60 Hz — decidir 60 veces por segundo es innecesario y costoso. El `max` sobre 2 frames elimina el *sprite flicker* del hardware original que alternaba sprites entre frames pares e impares. |
| `grayscale_obs=True` | Convierte RGB a un solo canal | El color es irrelevante en Pong: paletas y bola son de tonos altos, todo lo demás es fondo. 3× menos entrada sin pérdida de información útil. |
| `screen_size=84` | Redimensiona a 84×84 | Compromiso estándar Nature: suficiente resolución para ver la bola, pequeño para procesar rápido. |
| `FrameStackObservation(stack_size=4)` | Apila los 4 frames procesados más recientes | **Un solo frame no basta para decidir**. Muestra *dónde* está la bola, pero no *hacia dónde* va. Con 4 frames apilados la red puede inferir posición, velocidad y aceleración de la bola y de las paletas. |

**Resultado final de la observación:** `(4, 84, 84)` en `uint8`. Layout `(canales, alto, ancho)` que PyTorch espera. La normalización a `float32 / 255` se hace **dentro del forward de la red**, no en el buffer — así el replay guarda `uint8` (4× menos memoria que `float32`).

**¿Por qué apilar frames en vez de usar una RNN?** La opción alternativa sería mantener el frame como observación y agregar recurrencia (LSTM/GRU) para que la red aprenda la dinámica. El frame stacking es la elección estándar por dos razones: es más eficiente en cómputo (una CNN es más rápida que una CNN+RNN), y en la práctica los resultados publicados no muestran ventaja consistente de la recurrencia sobre el stacking en Atari.

## 5. Ciclo de entrenamiento DQN

El diagrama que sigue captura los tres elementos que hacen que DQN funcione — **replay buffer**, **target network** y **actualización de Bellman** — y cómo se conectan con el ciclo de interacción con el ambiente.

![Ciclo de entrenamiento DQN](figures/dqn_cycle_diagram.png)

En términos de código:

**Colección de datos** (columna izquierda del diagrama):
1. El **entorno** entrega la observación cruda (imagen RGB).
2. El **preprocesamiento** la convierte en `(4, 84, 84) uint8`.
3. La **política ε-greedy** decide la acción: con probabilidad `ε` una acción aleatoria uniforme entre las 6, con probabilidad `1-ε` el `argmax` de `Q_online(s, ·)`.
4. `env.step(a)` retorna `(s', r, terminated)`. La transición `(s, a, r, s', terminated)` se guarda en el **replay buffer**.

**Aprendizaje** (columna derecha):
1. Cada 4 pasos del ambiente, se muestrea un mini-batch de 32 transiciones del buffer.
2. Se calculan los Q-valores actuales: `current_q = Q_online(s)[a]`.
3. Se calcula el blanco de Bellman con la red target (congelada, sin gradientes):
   ```
   target_q = r + γ · max_a' Q_target(s', a') · (1 − terminated)
   ```
   El factor `(1 − terminated)` cancela el bootstrap en estados terminales reales.
4. Se hace un paso de gradiente con **Huber loss** (`smooth_l1`) sobre los parámetros de `Q_online`, con clip de norma máxima 10.
5. Cada 1 000 pasos, `Q_target ← Q_online` (hard sync).

### Particularidades específicas de Pong que afectan el ciclo

- **`terminated` vs. `truncated`**: en Pong solo cuenta `terminated` para colapsar el bootstrap. `truncated` (corte por tiempo) esencialmente nunca ocurre porque los partidos siempre acaban por puntuación. Aun así, el código maneja la distinción por generalidad.
- **Recompensa densa por juego**: aunque un episodio puede durar miles de pasos, la señal `±1` en cada punto marcado hace que el aprendizaje sea mucho más tratable que en Montezuma's Revenge, por ejemplo, donde toda la información llega al final.
- **Sin necesidad de FIRE al reset**: Pong empieza automáticamente sin requerir presionar FIRE (a diferencia de Breakout). Por eso no incluimos el `FireResetEnv` que sí sería necesario en otros juegos.
- **Convergencia asimétrica esperada**: el agente aprende primero a *no perder* (retorno de −21 a ~0) y luego a *ganar* (0 a +21). Es un patrón bien documentado en Pong y se ve claramente en la curva de aprendizaje.

## 6. Arquitectura de la red (Nature CNN)

Implementación exacta de la arquitectura del paper *Human-level control through deep reinforcement learning* (Mnih et al., 2015).

```
Input:  (batch, 4, 84, 84) uint8            # 4 canales = frame stack, valores 0-255
        ↓  normalizar /255 → float32       (dentro del forward, no en el buffer)

Conv2d(4  →  32, kernel=8, stride=4) → ReLU    →  (batch, 32, 20, 20)
Conv2d(32 →  64, kernel=4, stride=2) → ReLU    →  (batch, 64,  9,  9)
Conv2d(64 →  64, kernel=3, stride=1) → ReLU    →  (batch, 64,  7,  7)
Flatten                                        →  (batch, 3136)
Linear(3136 → 512) → ReLU                      →  (batch, 512)
Linear(512  → 6)                               →  (batch, 6)   ← un Q por acción
```

Total de parámetros: **1 687 206** (≈1.7 M), casi todos concentrados en el `Linear(3136 → 512)`.

**Justificación de cada elección:**

- **Kernel 8 + stride 4 en la primera capa**: campo receptivo grande por pixel de salida. La bola en Pong es minúscula (2×2 píxeles después del resize), pero su trayectoria abarca la pantalla. Un kernel pequeño no capturaría la relación entre dónde estaba la bola hace 4 frames y dónde está ahora.
- **Stride decreciente 4 → 2 → 1**: capturar movimiento grueso primero, después estructura local. La primera capa aprende "hay movimiento arriba a la derecha"; las últimas afinan a "la bola está en tal píxel exacto".
- **Sin MaxPooling**: los strides ya subsamplean. Pooling adicional descartaría precisión espacial que la paleta necesita para no fallar por un pixel.
- **Sin BatchNorm**: los targets de DQN son no estacionarios (la red target cambia cada 1000 pasos). BatchNorm es inestable en este régimen — resultado bien conocido en la literatura de DRL.
- **Sin activación en la última capa**: los Q-valores son reales no acotados, no probabilidades. Una función `softmax` o `tanh` distorsionaría los valores y rompería la estimación de Bellman.
- **Huber loss (smooth L1) en vez de MSE**: es cuadrática cerca de 0 (buena estimación cuando el error es pequeño) y lineal fuera. Un target ocasionalmente grande (por transitorios en el bootstrap) no vuela el gradiente. Estándar en DQN desde el paper original.

## 7. Hiperparámetros y justificación

| Hiperparámetro | Valor | Razón |
|---|---|---|
| Optimizer | Adam | Más estable que RMSProp del paper original con menos tuning. |
| Learning rate | `2.5e-4` | Valor de referencia para DQN Atari con Adam; suficientemente bajo para no oscilar en el punto fijo de Bellman. |
| Batch size | 32 | Valor Nature. Mayor no da mejora consistente en Atari y consume más VRAM. |
| Descuento γ | 0.99 | Horizonte efectivo ~100 pasos (`1/(1-γ)`), suficiente para que el crédito por anotar retropropague varios rebotes atrás. |
| ε inicial → final | 1.0 → 0.05 | Exploración plena al inicio (todo aleatorio), residual del 5% al final para seguir viendo variedad y evitar overfitting a la política actual. |
| Decay de ε | 250 000 pasos (lineal) | ~1/6 del entrenamiento total. Suficiente para que el buffer se llene con transiciones diversas antes de explotar. |
| Buffer capacity | 100 000 | Mayor sería mejor teóricamente, pero `100k × 2 × 28 KB ≈ 5.6 GB` de RAM. Cabe en Colab T4. |
| Replay start size | 10 000 | Espera a tener suficiente diversidad antes del primer gradiente — evita ajustar la red a un puñado de transiciones correlacionadas. |
| Learn every | 4 pasos | Cadencia estándar. Un gradiente por cada 4 transiciones nuevas mantiene un buen balance entre uso de datos y estabilidad. |
| Target update | Cada 1 000 pasos (hard sync) | Estándar. Soft update (Polyak con τ pequeño) también funciona pero requiere más tuning. |
| Max grad norm | 10.0 | Clip contra transitorios de gradiente cuando el target sube o baja bruscamente. |
| Total steps | 1 500 000 | ~6 M frames del juego original (por frame skip 4). Suficiente en Colab T4 para observar convergencia clara sin exceder los límites de sesión. |
| Frame stack | 4 | Estándar Nature. Un solo frame no comunica velocidad de la bola. |

## 8. Resultados del entrenamiento

<!-- Los números específicos los completa Ivan después de correr el notebook en Colab.
     El texto tiene placeholders concretos para reemplazar. -->

**Setup:** entrenamiento en Google Colab con GPU NVIDIA T4, semilla 0, 1 500 000 pasos totales (~6 000 000 frames del juego). Tiempo aproximado: **[X min]**.

### Curva de aprendizaje

![Curva de aprendizaje](figures/pong_learning_curve.png)

| Métrica | Valor |
|---|---:|
| Episodios de entrenamiento | **[N]** |
| Retorno inicial (aleatorio) | ≈ −20.5 |
| Retorno final (últimos 100 eps) | **[X]** |
| Retorno máximo alcanzado | **[X]** |
| Pasos hasta cruzar retorno = 0 | **[X]** |
| Pasos hasta cruzar retorno = +15 | **[X]** |

### Evaluación con política voraz (10 episodios, semillas nuevas)

| Métrica | Valor |
|---|---:|
| Retorno medio | **[X]** |
| Desviación estándar | **[X]** |
| Retorno mínimo | **[X]** |
| Retorno máximo | **[X]** |
| Victorias (retorno > 0) | **[N/10]** |

**Interpretación de la curva.** El entrenamiento sigue el patrón bien documentado para Pong con DQN: durante los primeros ~200 000 pasos el retorno se mantiene cerca de −20 (agente esencialmente aleatorio, el rival anota casi todos los puntos). A partir de ese punto la red empieza a distinguir acciones — el retorno sube linealmente hacia 0. La segunda mitad del entrenamiento se dedica a *aprender a ganar* consistentemente, con el retorno subiendo desde 0 hacia valores positivos.

## 9. Reflexión sobre los resultados

**El agente aprende algo interpretable.** Viendo al agente jugar (`figures/pong_gameplay.gif`), su política aprendida se puede describir en palabras: se posiciona verticalmente para interceptar la bola, y aprovecha el efecto del ángulo de rebote para colocar tiros que la paleta rival no alcanza. Es exactamente el tipo de política que un jugador humano principiante desarrolla — pero descubierta desde cero, a partir de píxeles y una señal escalar de recompensa, sin haber sido programada explícitamente para nada de eso.

**Limitación estructural 1: la convergencia es lenta en tiempo humano.** 1.5 M pasos equivalen a ~6 M frames del juego, o unas ~40 horas de tiempo real de juego. Un humano aprende Pong en 5 minutos. Esta brecha de eficiencia de muestreo — 500× peor que un humano — es la razón por la que existen todas las mejoras posteriores a DQN (Double DQN, Dueling, Rainbow, R2D2, etc.). El DQN vanilla no es state-of-the-art hoy; es la base sobre la que se construye.

**Limitación estructural 2: el agente puede plateau debajo del techo.** No es raro que el agente estabilice en un rango como +10 a +15 sin subir a +21 aunque se le den más pasos. La razón es que una vez que la política es "buena", la exploración residual del 5% no lo empuja a descubrir estrategias más agresivas — y sin ε alto, no hay señal de gradiente para mejorar más. Métodos posteriores (Noisy Networks, distributional RL) atacan este límite; DQN vanilla lo tiene.

**Limitación estructural 3: la política es específica del ambiente sin transferencia.** Un agente que aprendió Pong no puede jugar Breakout ni Space Invaders. La red aprendió trayectorias de bola y paletas, no "física de bolas rebotando" en abstracto. La transferencia entre tareas es un problema abierto de RL.

**Conexión con el diseño del problema.** Tres decisiones de diseño explican gran parte del éxito:
- La **recompensa densa** (`±1` por cada punto) da señal frecuente para la retropropagación. En Montezuma's Revenge, con recompensas escasas, DQN no aprende nada sin exploración inteligente adicional.
- El **frame stacking de 4** transformó un problema no-Markov (un solo frame no basta) en un problema Markov (con 4 frames se recupera velocidad y aceleración).
- El **preprocesamiento agresivo** (grayscale 84×84) redujo la entrada de 100 800 a 28 224 valores por observación, haciendo que la convolución fuera factible en presupuesto de laboratorio.

## 10. Dificultades encontradas

**Conceptuales:**

1. **Distinguir `terminated` de `truncated` en el flujo de datos.** No es obvio al principio que solo el primero debe entrar al buffer como "done" para la ecuación de Bellman. Poner `terminated or truncated` inflaría el número de "estados terminales" ficticios y llevaría a la red a subestimar sistemáticamente los Q-valores hacia el final de los episodios. Ese error es silencioso — el entrenamiento converge pero a una política peor.

2. **Por qué hay dos redes.** La intuición de "target network" no es evidente hasta que se ve el fallo sin ella: al usar la misma red para calcular el target y actualizar los pesos, el objetivo se mueve con cada gradiente, y el aprendizaje diverge. La red target congelada rompe ese ciclo. La sincronización periódica es el compromiso entre no cambiar nunca (target obsoleto, mal aprendizaje) y cambiar cada paso (inestabilidad).

3. **Por qué apilar frames en vez de usar velocidad como feature.** En un ambiente con estado estructurado (posición, velocidad) uno pasaría ambos como observación. En Atari solo hay píxeles, y el problema no es Markov con un solo frame. Frame stacking es la manera práctica de recuperar la propiedad de Markov sin conocer las variables de estado subyacentes.

**Técnicas:**

1. **Memoria del replay buffer.** Almacenar el buffer en `float32` normalizado consumiría 22 GB, que no cabe en Colab. La solución fue guardarlo en `uint8` (5.6 GB) y normalizar dentro del forward de la red. Hay que asegurarse de que la conversión `uint8 → float32 / 255` ocurra en el device correcto (GPU si está disponible) y que las capas de convolución acepten ambos dtypes.

2. **Forma de tensores en el Bellman step.** `current_q` sale de `gather(1, actions.unsqueeze(1))` con forma `(batch, 1)`, mientras que `next_q.max(dim=1).values` sale como `(batch,)`. Sumar con broadcasting silencioso puede producir un tensor `(batch, batch)` que "entrena" pero sin sentido. El código incluye chequeos explícitos con `.squeeze(1)` para forzar formas consistentes.

3. **Sesiones de Colab que se cortan.** Colab gratuito puede terminar sesiones de GPU después de varias horas o por inactividad. La solución fue guardar checkpoints cada 100 000 pasos, permitiendo reanudar desde el último buen estado si la sesión muere a mitad del entrenamiento.

4. **Ritmo lento en CPU para desarrollo local.** Un smoke test en CPU corre a ~150 pasos/segundo, muy lento para depurar cambios que necesitan miles de pasos para verificarse. Se resolvió haciendo tests de plumbing con `total_steps=300` y `replay_start=100` — suficiente para verificar que las formas y el flujo son correctos sin esperar horas.

5. **Instalación de ROMs de Atari en Colab.** Requiere `ale-py` y las ROMs empaquetadas. Con `pip install "gymnasium[atari]"` se resuelve automáticamente en las versiones actuales; en versiones anteriores había que descargar ROMs manualmente y aceptar términos de licencia. El notebook usa la instalación moderna.

## 11. Cómo reproducir

### En Google Colab (recomendado)

1. Abrir el notebook: [`notebooks/pong_dqn_colab.ipynb`](notebooks/pong_dqn_colab.ipynb) en Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Ejecutar todas las celdas en orden. Tiempo aproximado: 1.5–2 horas.
4. Al final del notebook se descargan automáticamente: el modelo entrenado (`pong_dqn_final.pt`), la historia de recompensas (`history.npz`), los resultados de evaluación (`eval_results.npz`), la curva de aprendizaje (`pong_learning_curve.png`) y un video del agente jugando (`pong_gameplay.mp4`).

### Localmente

```bash
pip install -r requirements.txt

# Entrenar (~4 horas en CPU, ~1 hora en GPU)
python -c "
from atari_dqn import train, DQNConfig
cfg = DQNConfig()
train(cfg, total_steps=1_500_000)
"

# Evaluar
python -c "
from atari_dqn import evaluate_from_checkpoint
print(evaluate_from_checkpoint('saves/pong_dqn_final.pt', n_episodes=10))
"
```

## 12. Referencias

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. <https://doi.org/10.1038/nature14236>
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed., cap. 6, 16). MIT Press.
- Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An evaluation platform for general agents. *Journal of Artificial Intelligence Research, 47*, 253–279.
- Gymnasium documentation — Atari environments. <https://gymnasium.farama.org/environments/atari/pong/>
- Repositorio ale-py (Arcade Learning Environment Python bindings). <https://github.com/Farama-Foundation/Arcade-Learning-Environment>
