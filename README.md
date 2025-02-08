# challange-multi-task-change-detection
Detecting change in multi-image, multi-date remote sensing data helps discover and understand global conditions. This challenge uses geographical features obtained from satellite images. Computer vision techniques have been used to process the data that is now ready to be explored using machine learning methods.

## Features
The geographical features of the data are:

- An irregular polygon.
- Categorical values describing the status of the polygon on five different dates (for example, the polygon was under construction on day 1, and construction was completed on the following four dates).
- Neighbourhood urban features (e.g., the polygon is in a dense urban and industrial region).
- Neighbourhood geographic features (e.g., the polygon is near a river and a hill).

## Pipeline
The proposed pipeline is the one introduced in the first lecture of the ML course and is similar to the ones seen in the labs. The skeleton_code.py python script implements a simple k-NN baseline model, achieving performance close to 40%.

Data preprocessing: You must preprocess the data to convert it into an appropriate format.

Feature engineering and dimensionality reduction: Explore creating a new (smaller) set of features from existing ones. Is it possible to improve performance using dimensionality reduction techniques? Moreover, is it beneficial to select a subset of the original features (feature selection) for this task? Here are some ideas to help you with feature engineering:

Urban and Geographical types are multi-valued categorical columns (one hot encoding could be helpful in this case).
Irregular polygons can be processed in several ways to create features: the area, the perimeter, or any other geometrical property of a polygon.
The number of days between two consecutive dates could also be helpful.
Learning algorithm: Next, you will choose an appropriate learning (in this case, classification) algorithm to solve the problem. What are the characteristics of the problem and data (type of task, is it supervised, dimensionality, is there noise)? Is logistic regression the best choice? Maybe an SVM, decision trees, or even neural networks would perform better? Finally, is it worth implementing an ensemble learning strategy and combining multiple classification algorithms?

Evaluation: Your final choice depends on the evaluation metric. Remember that you are solving a multi-class classification task, and different metrics have different meanings! Finally, what can you say about the generalization of your model? More details about the evaluation are provided in the next section.

Evaluation
You will build your classification model based on the training data in the train.geojson file and test data in the test.geojson file. As we have seen in the course, cross-validation is a helpful technique (see https://scikit-learn.org/stable/modules/cross_validation.html) for supervised problems. Recall that the final evaluation of your model is based on the test dataset contained in the test.geojson file.

## Evaluation metric

The evaluation metric for this competition is the Mean F1-Score. The F1 metric weights recall and precision equally, meaning a good classification model under this metric will maximize both. An average performance for both criteria is favoured over excellent performance for one criterion and poor performance for the other.

## How do you evaluate your model on the test dataset?

First, note that you must preprocess the test data like you have preprocessed the training data. For the test data, we do not have information about the class labels (type of area); therefore, the final assessment will be done on the Kaggle platform. The evaluation process can be summarized as follows:

Run your model on the test data (test.geojson).

Get the predicted class labels for each instance of the test dataset.

Create a new file, sample submission.csv, containing the results (the predicted class label for each instance). The file must contain a header and have the following format (an example file is provided in the Data section of the challenge; see knn_sample_submission.csv).

The dataset is part of the paper https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html

nouzir. 2EL1730 Machine Learning Project - Jan. 2025. https://kaggle.com/competitions/2-el-1730-machine-learning-project-jan-2025, 2025. Kaggle.