# Visual-Category-Recognition
An Image Recognition system which makes use of kNN and SVMs for prediction.

# Platform used:
Programming language: Python 2.7 | IDE: jupyter-notebook | libraries: scikit-image, scikit-learn

# Running the code:
Hybrid+Classifier.py file can be executed through terminal. This script calls function make_prediction() whose parameters can be changed to make prediciton for some other image in dataset(assuming you have downloaded the data). Other notebooks are there to be refered for heatmap and confusion matrix of prediciton made by our hybrid model.

# What was the motive for doing this project?
This project do not have any significant practical use, we(me and one of my classmate) build this project as MINOR project for my 5th semester. The purpose for this project was to build a fundamental base in the field of Computer Vision, so that the knowledge gained by doing this project can be applied for our major project which will definitely be ready for a commerial aplication. Our Major project will making use of RNNs and CNNs (deep learning) to generate natural language description for an image and we will soon be adding a repository to that project.

This doesn't mean we didn't put any effort building this object recognition system. It took a signigficant time to complete this proect as the domain of Computer Vision was completely unknown to us and we spent a lots of time reading research papers and learning theory behind Support Vector Machines.

# Technologies used:

## Image Processing:
PCA - Principal Component Analysis

## Machine Learning:
kNN - k-nearest neighbour algorithm | SVMs - Support Vector Machines

# Notebooks:
- SVM Classifier.ipynb: Contains code for classifier solely trained on Support Vector Machines (bad Accuracy) and Confusion Matrix representation of results.
- Hybrid Classifier.ipynb: Classifier making use of kNN algorithm on top of SVM adding another filtering layer to improve accuracy. (improved accuracy)
- FinalProjectreport: Report submitted by us in college.
- RefferedResearchPaper.pdf: Algorithm that we implemented in this project was based on this research paper.

# What's the best thing about this project?
- Best thing about this project is that the hybrid classfier is trained in a completely different way as the classifier solely based on SVMs. And the improvement in accuray is considerable. Use of kNN makes the classifier SCALABLE.
- We were able to get 29% of accuracy on 10 classes without using any neural networks.
