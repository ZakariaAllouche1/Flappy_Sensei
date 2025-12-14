# Flappy Sensei – Imitation Learning pour Flappy Bird

Projet de Master 2 sur l’**apprentissage par imitation** appliqué au jeu **Flappy Bird**.

Objectifs principaux :
- Apprendre une politique à partir de **démonstrations humaines** (Behavioral Cloning).
- Améliorer cette politique avec **DAgger** (corrections interactives).
- Comparer avec des agents **RL** (PPO, DQN – Stable-Baselines3) sur le même environnement visuel.

---

## 1. Structure du projet

```text
.
├─ data/                    # Démonstrations humaines et données DAgger (.npz)
├─ flappy-bird-env/         # Code d’enregistrement de l'env Gymnasium FlappyBird-v0
├─ models/                  # Modèles entraînés
│  ├─ bc_student_human.pth
│  ├─ bc_student_human_v2.pth
│  ├─ bc_student_human_dagger1.pth
│  ├─ bc_student_human_v2_dagger1.pth    # modèle BC+DAgger final
│  ├─ expert_dqn_flappy_v3.zip           # agent DQN Stable-Baselines3
│  └─ expert_ppo_flappy_v1_framestack4.zip  # agent PPO Stable-Baselines3
├─ scripts/                 # Scripts principaux
│  ├─ collect_human_demos.py
│  ├─ dagger_collect_human_iter1.py
│  ├─ demo_eval_BC_dagger.py
│  ├─ demo_eval_RL.py
│  ├─ train_bc_human.py
│  ├─ train_bc_human_v2.py
│  ├─ train_bc_human_dagger1.py
│  ├─ train_bc_v0.py             
│  ├─ train_expert_dqn_v3.py
│  ├─ train_expert_ppo_v1.py
│  └─ train_flappy_ppo.py        
├─ requirements.txt
└─ README.md
```

---

## 2. Installation

### 2.1. Créer et activer un environnement virtuel

Depuis la racine du projet :

```bash
python -m venv .venv
```

**Activation :**

* Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

* Linux / macOS
```bash
source .venv/bin/activate
```

### 2.2. Installer les dépendances

Une fois l'environnement virtuel activé :

```bash
python -m pip install -r requirements.txt
```