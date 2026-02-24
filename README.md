# <img src="./docs/images/ftsts-logo.png" alt="Logo" width="50" style="vertical-align: middle; margin-right: 10px;"> NeuroLoop

_Closed-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

[docs](https://owenmastropietro.github.io/projects/neuroloop/)

---

<p align="center">
    <img src="./docs/images/neuroloop-system-overview.png" alt="ftsts-logo" width="300"/>
  <br>
  <em>The overall architecture of this work.</em>
</p>

**NeuroLoop** implements a closed-loop deep brain stimulation (cl-DBS) system to modulate neural synchronization in a computational model of the brain, optimizing the open-loop regime described in (_[Schmalz & Kumar, 2019](https://doi.org/10.3389/fncom.2019.00061)_) by implementing a Reinforcement-Learning (RL) driven feedback controller that adjusts stimulation parameters based on real-time measurements of network synchronization.

Pathological neural synchronization is a fundamental characteristic of several neurological and neuropsychiatric disorders, including Parkinson’s disease, epilepsy, and major depressive disorder, where excessive coupling between neural populations can directly contribute to debilitating symptoms such as tremor, seizures, and cognitive impairment.
Deep brain stimulation ([DBS](https://en.wikipedia.org/wiki/Deep_brain_stimulation)) is an established therapy for modulating abnormal activity, but current clinical systems often operate in an open-loop manner, delivering stimulation continuously without adapting to the brain’s evolving state.

---

## Usage

> Note: you must first clone [dbsenv](https://github.com/ftsts/dbsenv) for this project.

```sh
# Install the environment (e.g.,)
pip install --editable ../dbsenv

# Run a script (e.g.,)
nl run --open

# Visualize training
tensorboard --logdir ./ppo_tensorboard
```

## Related Repositories

- **[ftsts-ol](https://github.com/ftsts/ftsts-ol)** - implementation based on original open-loop regime.
- **[dbsenv](https://github.com/ftsts/dbsenv)** - custom OpenAI Gym environment for simulating closed-loop DBS.

## Literature

- **[ftsts](https://doi.org/10.3389/fncom.2019.00061)** - original work on open-loop stimulation regime.
- **[ftsts _(post-ictal)_](https://doi.org/10.3389/fncom.2023.1084080)** - extension of original work to target post-ictal desynchronization.
- **ftsts _(pi)_** - extension of original work to close the loop using proportional-integral (PI) control. _(contact for material)_
- **[rl driven cl-dbs](https://doi.org/10.1109/tnsre.2024.3465243)** - a study on using reinforcement learning agents for adaptive parameter control in a closed loop DBS environment based on the basal ganglia-thamalic (BGT) model.
