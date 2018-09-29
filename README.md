# Face Orientation and Emotion Classification

The goal of the project was to create a software that would recognise and classify the orientation of the face and the emotion of 70 different subjects (males and females). The dataset is taken from the Karolinska Directed Emotional Faces (KDEF)* and comprises 4,822 photos taken at different angles (frontal, half profile and full profile) showing 7 emotions (afraid, angry, disgusted, happy, neutral, sad, surprised).
Labels of the dataset (train and test) are provided in the *dataset* folder.
The *src* folder contains:
* *SplitDataset.py* : splits the dataset into training and test set and creates the respective csv files;
* *preprocessing.py* : preprocesses the photos using contrast stretching and gamma enhancement;
* *face_detection.py* : converts the photos from RGB to HSV and uses the Value component to detect the face;
* *feature_detection.py* : adds a padding to all images in order to have pics of the same dimension, includes implementations of Linear Discriminant Analysis and Histogram of Oriented Gradients;
* *Pipeline.py* : calls functions from other modules to implement the train or test pipeline (*see SL_Report.pdf*) and classifies the Orientation of the Face using the Histogram of Oriented Gradients + Support Vector Machines, and then passes the output of the classification to the Emotion Classification step comprising the Linear Discriminant Analysis + Support Vector Machines
* *main.py* : pipelines of training and test set


#### for more information, see SL_Report.pdf


*Lundqvist, D., Flykt, A., & Öhman, A. (1998). The Karolinska Directed Emotional Faces – KDEF, CD ROM from Department of Clinical Neuroscience, Psychology section, Karolinska Institutet, ISBN 91-630-7164-9.
