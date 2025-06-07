# CS188 Final Project: Imitation Learning for Square Assembly

This repository contains the final project for UCLA's CS188: Introduction to Robotics (Spring 2025). The project focuses on solving the Robosuite Square Assembly task using imitation learning techniques.

## 📌 Project Overview

The objective is to train a robotic manipulator to perform the Square Assembly task by learning from demonstrations. Two primary approaches were implemented:

* **Dynamic Movement Primitives (DMP):** A trajectory-based method that models movements using dynamical systems.
* **Multi-Layer Perceptron (MLP):** A neural network-based behavioral cloning approach that maps observations directly to actions.

## 📁 Repository Structure

```
CS188_Final_Project/
├── DMP/                     # Implementation of the DMP approach
├── mlp/                     # Implementation of the MLP approach
├── demos.npz                # Dataset containing demonstration trajectories
├── load_data.py             # Script for loading and preprocessing data
├── README.md                # Project documentation
```

## 🧠 Approaches

### Dynamic Movement Primitives (DMP)

The DMP approach models the demonstrated trajectories as a set of differential equations, allowing for generalization to new goals and robustness to perturbations. It captures the essential shape of the movement while being flexible to changes in the task parameters.

### Multi-Layer Perceptron (MLP)

The MLP approach utilizes a feedforward neural network trained via supervised learning. It takes in the robot's observations and outputs the corresponding actions, effectively cloning the behavior demonstrated in the dataset.

## 🛠️ Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SanjayKrishnas/CS188_Final_Project.git
   cd CS188_Final_Project
   ```


3. **Run the MLP Approach:**

   ```bash
   cd mlp
   python train.py
   python test.py
   ```

4. **Run the DMP Approach:**

   ```bash
   cd DMP
   python dmp_policy.py
   ```


## 👥 Contributors

* **Sanjay Krishna** 
* **Nakul Joshi** 

