# Predicting Events based on Offline-Data Collected with the ANYmal Research Robot

This is part of my course project for the course Foundations of Reinforcement Learning at ETH.

## How to run the Code?

1. Clone the repository
2. Install the requirements
3. Save the data inside the `data` folder and adjust the paths in `main.py` in order to load the correct `.bag`files.
4. Run the `main.py` file

This will pre-process the data, then saves it as a `.pkl` file (which can be loaded later) and then trains the model.
Pre-processing the data is a time-consuming process.

## What is the project about?

### Abstract

> We investigate using offline reinforcement learning to enhance the navigation
> capabilities of legged robots using previously collected real-world data. Our
> approach utilizes a comprehensive dataset from past ANYmal missions to develop
> a helper component for navigation that predicts the likelihood of navigational
> events based on sensor inputs.

This project is part of the course project for the course Foundations of Reinforcement Learning at ETH.