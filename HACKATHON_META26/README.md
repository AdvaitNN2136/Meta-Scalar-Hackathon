---
title: Disaster Response Management
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
 - openenv
---
# Disaster Response Management System

A reinforcement learning environment where an agent must coordinate
rescue teams across 8 Indian cities facing cascading disasters.

## Environment Description
Real-world disaster response simulation with cascade effects:
- Earthquake triggers Tsunami in coastal cities
- Lightning triggers compound fires
- Flood causes infrastructure collapse and epidemic
- Day/Night cycle affects team effectiveness
- Power grid failures cascade across connected cities

## Cities
Chennai, Mumbai, Delhi, Bengaluru, Mysuru, Hyderabad, Kochi, Coorg

## Observation Space
Box(85,) — float32
- Per city (8 x 10): severity, disaster_type, compound_flag,
  infrastructure_dmg, power_out, evacuation_status, disease_risk,
  team_fatigue, population_density, elevation
- Global (5): teams_available, supply_level, time_of_day,
  step_fraction, weather_state

## Action Space
Discrete(18)
- 0: Hold
- 1-8: Dispatch team to city N
- 9-16: Evacuate city N
- 17: Rest teams (recover fatigue)

## Tasks
| Task | Difficulty | Objective | Threshold |
|---|---|---|---|
| task_1_contain | Easy | Keep avg severity < 0.5 | 0.6 |
| task_2_prevent_cascade | Medium | < 10 cascade events | 0.7 |
| task_3_mass_casualty | Hard | Minimize casualties in compound disasters | 0.8 |

## Setup
pip install -r requirements.txt

## Run inference
python inference.py

## Run baseline graders
python tasks.py

## Run Gradio demo
python app.py

## Docker
docker build -t disaster-response .
docker run -p 7860:7860 disaster-response

## Baseline Scores
- task_1_contain: ~0.72
- task_2_prevent_cascade: ~0.65
- task_3_mass_casualty: ~0.58
