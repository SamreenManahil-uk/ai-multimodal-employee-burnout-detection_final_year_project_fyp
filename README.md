# 🏆 Flagship Master's Project — AI-Based Multimodal Employee Burnout Detection System

**University-Recognised Research | Machine Learning | NLP | Computer Vision | Explainable AI**

**Selected as one of the university's best Master's projects.**

## 🌟 Project Overview

This flagship Master's project presents an innovative, explainable multimodal Artificial Intelligence system designed to assess workplace burnout risk by integrating three powerful AI technologies: Machine Learning, Natural Language Processing (NLP) and Computer Vision.

Unlike conventional approaches that rely primarily on employee surveys or a single source of information, this system analyses multiple types of employee-related information to produce a comprehensive burnout risk assessment.

The project addresses a significant research gap. A critical review of 17 relevant studies identified that none of the reviewed studies combined all three AI modalities, explainable predictions and a functional application within one integrated framework.

The result is an ambitious research prototype demonstrating how advanced AI technologies can work together to support employee wellbeing.

## 🎯 What Does This System Do?

The system assesses potential employee burnout by combining three independent sources of information.

**1. Machine Learning — Workplace Burnout Prediction**

The machine-learning component analyses structured workplace and wellbeing information, including:

- Employee stress levels
- Weekly working hours
- Work-life balance
- Mental health indicators
- Workplace support
- Lifestyle-related factors

Random Forest and XGBoost algorithms were developed and compared to identify patterns associated with employee burnout risk.

The final XGBoost model achieved **92.77% test accuracy and a 0.9580 ROC-AUC** on the structured-data test set.

**2. Natural Language Processing — Employee Feedback Analysis**

The NLP component uses a pretrained DistilBERT model to analyse written employee feedback.

For example, an employee might write:

"I'm constantly exhausted, struggling with my workload and finding it difficult to maintain a healthy work-life balance."

The NLP component evaluates the text against categories including emotional exhaustion, workplace stress, job dissatisfaction and positive workplace wellbeing.

This provides an additional signal that structured workplace data alone might not capture.

**3. Computer Vision — Facial Emotion Analysis**

The Computer Vision component uses DeepFace to analyse facial expressions and identify emotional categories, including sadness, happiness, anger, fear and neutrality.

These observations provide supplementary information that can be considered alongside structured data and written feedback.

Importantly, facial expressions alone cannot reliably establish whether an employee is experiencing burnout.

**4. Multimodal AI — Combining All Three Technologies**

The project's central innovation is its multimodal fusion architecture.

Instead of generating three unrelated predictions, the system combines information from Machine Learning, NLP and Computer Vision into one integrated risk score.

The weighted fusion approach assigns:

- Machine Learning: 50%
- Natural Language Processing: 30%
- Computer Vision: 20%

This architecture demonstrates how complementary AI components can be integrated into a unified decision-support framework.

**5. Explainable AI — Understanding Predictions**

The system incorporates SHAP to explain the machine-learning component's predictions.

Rather than simply presenting a burnout risk score, SHAP identifies the factors that influenced the prediction, such as stress levels, working hours and work-life balance.

This improves transparency and helps users understand the reasoning behind the structured-data model's output.

SHAP identifies statistical contributions to predictions rather than proving the actual causes of burnout.

## 💡 How Is This Project Beneficial?

The project explores how Artificial Intelligence could support organisations in identifying potential employee wellbeing concerns and improving workplace support.

**Early identification of potential burnout**

By analysing workplace-related indicators, the system could help highlight employees who may benefit from additional wellbeing support.

Identifying potential concerns earlier could allow organisations to offer appropriate assistance before problems become more serious.

**A more comprehensive assessment**

Traditional employee wellbeing assessments may rely heavily on periodic surveys or individual sources of information.

This project demonstrates how structured workplace data, written feedback and facial emotion analysis can be combined to provide a broader collection of signals.

However, the project does not establish that multimodal fusion improves real-world diagnostic accuracy over the individual components.

**Supporting informed HR decisions**

The system is designed as a potential decision-support tool rather than an automated employee evaluation mechanism.

With appropriate validation and safeguards, such technology could help HR professionals identify situations that warrant supportive conversations, workload reviews or additional wellbeing resources.

**Transparent and interpretable AI**

SHAP makes the structured-data model's predictions more understandable.

This is particularly important in workplace applications, where decisions involving employees should be transparent, fair and subject to human review.

**Potential organisational benefits**

With further validation and responsible implementation, the underlying approach could contribute to:

- More proactive employee wellbeing programmes.
- Earlier identification of potential workplace stress.
- Better-informed workload and wellbeing discussions.
- Research into responsible AI-assisted workplace support.
- Improved understanding of how different AI modalities interact.

These are potential applications rather than benefits established through real-world organisational trials.

## 🔬 Research Innovation and Technical Achievements

This project demonstrates substantial independent research and technical development across several specialised AI disciplines.

**Key achievements:**

- Developed a machine-learning pipeline using a dataset of 150,000 records, including feature engineering, class imbalance handling and systematic model evaluation.
- Compared Random Forest and XGBoost using five-fold cross-validation and F1-score-based model selection.
- Integrated pretrained DistilBERT for zero-shot workplace feedback classification.
- Incorporated DeepFace for facial emotion analysis.
- Designed a weighted late-fusion architecture combining three AI modalities.
- Applied SHAP to investigate feature importance and explain individual machine-learning predictions.
- Conducted fusion sensitivity experiments and diagnostic conflict-case testing.
- Developed a functional multimodal research application demonstrating the integrated architecture.

## 🏅 Academic Recognition and Research Contribution

**Recognised as one of the university's best Master's projects.**

The project's distinguishing contribution is its integration of multiple advanced AI technologies within one explainable, functional research prototype.

Among the 17 studies examined during the literature review, none demonstrated the complete combination of all three modalities, explainability and a working application.

This project was developed to address those identified gaps, demonstrating independent research, machine-learning engineering, multimodal system integration and critical evaluation.

The work also identifies important limitations, particularly the need for independently labelled datasets to validate the NLP and Computer Vision components.

The resulting prototype establishes a foundation for further research into multimodal AI, explainable predictions and responsible workplace wellbeing technologies.

**Project significance:** This flagship project demonstrates my ability to independently research, design, implement, integrate and evaluate complex Artificial Intelligence systems while addressing a meaningful real-world challenge through applied research and software engineering.