# CPSC322-Final Project

Danni Du & Lida Chen

Date: 12/11/2024

## Description:
The primary objective of this project is to analyze NEO data and build machine learning classifiers to predict whether an asteroid is hazardous based on its physical and orbital characteristics like the estimated size, velocity, distance, and magnitude of asteroids.

## Attributes:
(X)
1. est_diameter_min (Minimum Estimated Diameter in Kilometres)
1. est_diameter_max (Maximum Estimated Diameter in Kilometres)
1. relative_velocity (Velocity Relative to Earth)
1. miss_distance (Distance in Kilometres missed)
1. absolute_magnitude (Describes intrinsic luminosity)

(y)
1. hazardous (Boolean feature that shows whether asteroid is harmful or not)

(Others)
1. id (Unique Identifier for each Asteroid)
1. name (Name given by NASA)
1. orbiting_body (Planet that the asteroid orbits, Earth)
1. sentry_object (Included in sentry - an automated collision monitoring system, all false)

## Instructions to Run the Project:
  Step1: Clone the repo
  Step2: Run Unittest(test_myclassifiers.py)
    - Run pytest --verbose test_myclassifiers.py
  Step3: Connect Web App(website.py)
    - Run python website.py
    - Copy the link that shows on the terminal and paste it on the website

## Organize(analysis.ipynb):
1. Choosing data(Hazardous=1(5000),Hazardous=0(5000)
2. Fit data via KNN, Dummy, Naive Bayes, Decision tree, and Random Forest
3. Print information 
