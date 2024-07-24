# UNISA2023-24.Artificial_Vision Final Project

$\textsf{{\color[rgb]{0.0, 0.0, 1.0}A}{\color[rgb]{0.1, 0.0, 0.9}b}{\color[rgb]{0.2, 0.0, 0.8}s}{\color[rgb]{0.3, 0.0, 0.7}t}{\color[rgb]{0.4, 0.0, 0.6}r}{\color[rgb]{0.5, 0.0, 0.5}a}{\color[rgb]{0.6, 0.0, 0.4}c}{\color[rgb]{0.7, 0.0, 0.3}t}}$: This project revolves around creating a sophistical system capable of processing video inputs alongside a specified configuration, undertaking a sequence of pivotal operations. This includes the implementation of algorithms to accurately identify all individuals within a given scene, deployment of advanced tracking mechanisms to monitor and trace the movements of each individual over time, implement a pedestrian attribute recognition model for the acknowledgment of crucial attributes. Moreover, the project involves conducting in-depth analyses of the behavior of tracked individuals. This analysis
encompasses tracking their movements through predefined Regions of Interest (ROIs) and determining the duration of their presence and the total number of passages in these areas.

## 🛠 Testing
To execute the test, please use the following command:

```bash
python group08.py --video video.mp4 --configuration config.txt --results results.txt
```

![9](https://github.com/user-attachments/assets/4489ed7c-adef-488d-8d2b-b4b685d953ce)


## PAR (Pedestrian Attribute Recognition)

Pedestrian attributes recognition from images is nowadays a relevant problem in several real applications, such as digital signage, social robotics, business intelligence, people tracking and multi-camera person re-identification. To this concern, there is a great interest for recognizing simultaneously several information regarding pedestrians. The approach we propose leverages a **Multi-Task Attention Network** (**`MTANet`**), which is adept at handling the complexity and variability inherent in PAR tasks.

![5](https://github.com/user-attachments/assets/e37aa92b-bbdf-4078-908b-4e44fe3dd86f)
