# YAZD Dataset

# Central Courtyard Feature Extraction in Remote Sensing Aerial Images Using Deep Learning: A case-Study of Iran

**Abstract**

Central courtyards are one of the primary components of vernacular architecture in Iran. The central courtyards' direction, dimensions, ratio, and other characteristics are extremely valuable for studying as a historical passive cooling and heating solution. Several studies on central courtyards compare their features in different cities and climatic zones in Iran. In this study, deep learning methods for object detection and image segmentation are applied to aerial images to extract central courtyards' features. The case study is aerial images of nine historical cities in Bsk, Bsh, Bwk, and Bwh Köppen climate zones. Furthermore, These features are gathered in an extensive dataset with 26,437 samples and 76 geometric and climactic features. Additionally, the data analysis methods reveal significant correlations between various features such as the length and width of courtyards. In all cities, the correlation coefficient between these two characteristics is approximately +0.88. Numerous mathematical equations are generated for each city and climate zone by fitting the linear regression model to these data in different cities and climate zones. These equations can be used as proposed design models to assist designers and researchers in predicting and locating the best courtyard houses in Iran's historical regions.
#
The following are accessible via this page: 

The object detection and segmentation training codes consist of the final trained models for courtyard detection and segmentation, the final dataset, Figures, data analysis, and model fitting codes.
#
**List of files**

* **courtyard_detection_trainig.ipynb:** The training of the courtyard detection by the Remo_app onject detection tutorial
* **courtyard_segmentation.ipynb:** The training of the courtyard image segmentation by the Remo_app instance segmentation tutorial
* **courtyard_segmentation_offset.ipynb:**  The training of the courtyard_offset image segmentation by the Remo_app instance segmentation tutorial
* **courtyard_detection_crop.py:** The code that detects the central courtyards and crop them for two different datasets; courtyard and courtyard_offset datasets
* **Segmentation_to_dataframe.py:** The code that segment the objects in both datasets and extract the features of the object to create the main dataset. This code uses the weather.csv file to get the climate information of each sample and add these information to each sample. The output of the code is the main csv dataset file which is named dataset.csv.
* **data_analysis.ipynb:** Data cleaning and data analysis process on the main dataset that the samples of the outputs are several diagrams in the image folder.
* **dataset.csv:** The main dataset
* **dataset_2.csv:** The main dataset after removing the wrong and outlayers data.
* **dataset_geometric.csv:**  The final dataset with just geometrical features.
* **dataset_cities.csv:** The dataset of the mean of the features in different cities.
* **dataset_zones.csv:** The dataset of the mean of the features in different climatic zones.
