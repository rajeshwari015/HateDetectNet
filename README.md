# HateDetectNet: Hate Speech Detection using Convolutional and Bi-Directional GRU with Capsule Network
* HateDetectNet is a deep learning model designed for hate speech detection. It utilizes a combination of Convolutional Neural Networks (CNN), Bi-Directional Gated Recurrent Units (GRU), and Capsule Networks to achieve accurate hate speech classification.

## Features
* Utilizes a Convolutional Neural Network (CNN) for capturing local features in text sequences.
Employs a Bi-Directional Gated Recurrent Unit (GRU) to capture contextual information and handle long-term dependencies.
Integrates a Capsule Network to model hierarchical relationships between features and improve detection accuracy.
Provides an end-to-end hate speech detection solution.
Supports training on various datasets and can be fine-tuned for specific domains.

## Requirements
* Python 3.6
* TensorFlow (version X.X.X)
* Keras (version X.X.X)
* NumPy (version X.X.X)
* Pandas (version X.X.X)
* scikit-learn (version X.X.X)

## Usage
### Clone the repository:
* git clone https://github.com/rajeshwari015/hatedetectnet.git
### Install the required dependencies:

* pip install -r requirements.txt

## Prepare your dataset:

* Ensure your dataset is properly formatted for hate speech detection, with labeled examples.Split your dataset into training and testing sets.
Place the data files in the appropriate directories (e.g., data/train.csv, data/test.csv).

## Configure the model:

Adjust the hyperparameters, such as learning rate, batch size, and network architecture, in the config.py file.

## Train the model:

python train.py

## Evaluate the model:

python evaluate.py

## Results
Below are the performance metrics achieved by HateDetectNet on a standard hate speech detection dataset:

Accuracy: 0.85
Precision: 0.87
Recall: 0.83
F1-score: 0.85

## Contributing
Contributions to the HateDetectNet project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


## Acknowledgements
This project utilizes the power of TensorFlow and Keras deep learning frameworks.
We acknowledge the authors of the original datasets used in training and evaluation.
Special thanks to the open-source community for their valuable contributions.

## Contact
For any inquiries or questions, please contact rajeshwaripavuluri@gmail.com.


