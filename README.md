Multi-Modal Human Stress Detection
Project Overview
This project focuses on developing a comprehensive, multi-modal platform for accurate human stress detection. By integrating various data sources, we aim to provide a more complete and precise understanding of an individual's stress levels, moving beyond the limitations of traditional, single-source approaches.

The Problem
Traditional methods for stress evaluation often fall short due to their inability to account for individual differences and the complex, context-dependent nature of stress. This leads to an incomplete and potentially inaccurate assessment of an individual's true psychological state.

Our Multi-Modal Solution
We propose a robust platform that combines insights from three distinct yet complementary data sources:

Subjective Experiences: Captured through user self-report questionnaires.

Objective Cues: Derived from observable facial expressions.

Physiological Data: Obtained from bodily metrics.

By fusing these diverse data types, our system aims for a more comprehensive and precise assessment of psychological stress.

Key System Components & Features
Our platform integrates several core elements:

Chatbot-based Self-Report Questionnaires: An intuitive interface for users to report their subjective stress experiences.

Facial Expression Analysis: Utilizes computer vision techniques (e.g., face detection, facial landmark extraction, Action Unit (AU) analysis) to identify subtle facial movements indicative of stress.

Physiological Data Integration: Incorporates "Pythological Data" such as heart rate, respiration rate, body temperature, snoring range, and limb movement.

Machine Learning Models: Employs advanced deep learning architectures like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to fuse and analyze multi-modal data for stress prediction.

Streamlit Web Application: The central hub for data collection, storage, analysis, and visualization, providing a user-friendly interface.

Data & Initial Findings
Physiological Data Source: We utilize the "Stress Detection Dataset" from Kaggle.

Exploratory Data Analysis (EDA): Initial analysis revealed features with non-normality and the presence of outliers (e.g., snoring range around 71.6, respiration rate around 21.92, with some missing data for body temperature and limb movement). Preprocessing steps are applied to address these characteristics.

Preliminary Results
Our initial model training on physiological data has shown highly promising results:

Models Tested: XGBClassifier, LGBMClassifier, and CatBoostClassifier.

Training Accuracy: All models achieved 100% accuracy on the training data.

Test Accuracy: The CatBoostClassifier remarkably achieved 100% accuracy on the test set, suggesting strong predictive power for stress levels within our dataset. While this is excellent, we acknowledge the need to consider potential overfitting and dataset characteristics.

Project Objectives
Design and develop a robust multi-modal stress detection platform.

Integrate and fuse data from self-report questionnaires, facial expressions, and physiological measurements.

Effectively utilize machine learning models for accurate stress level prediction.

Develop a system capable of providing timely and contextually relevant stress assessments.

Expected olistic understanding of stress.
Technologies Used
Programming Language: Python

Web Framework: Streamlit

Machine Learning Libraries: Scikit-learn, XGBoost, LightGBM, CatBoost (and potentially TensorFlow/PyTorch for CNNs/RNNs)

Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

Computer Vision: Libraries for face detection, landmark extraction, and Action Unit analysis (e.g., OpenCV, Dlib - specific libraries to be confirmed during implementation)Outcomes & Impact

The primary outcome is a working prototype demonstrating the feasibility and effectiveness of leveraging AI and machine learning for comprehensive human stress detection. This research has the potential to enable personalized interventions and significantly improve individual well-being by offering a more accurate 
