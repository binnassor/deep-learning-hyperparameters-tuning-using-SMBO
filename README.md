# hyperparameters-tuning-using-SMBO
This repository implements Sequential Model-Based Optimization (SMBO) for hyperparameter tuning and image classification 
using four popular deep learning models: MobileNetV2, MobileNetV3-Small, ShuffleNetV2, and SqueezeNet. 
The project focuses on optimizing key hyperparameters to improve model performance, followed by model training and testing.

Table of Contents
•	Introduction
•	Models
•	Optimization Approach
•	Training Strategy
•	Dataset
•	Results
•	License

Introduction
Hyperparameter tuning is essential to achieve optimal performance for deep learning models. In this project, 
SMBO is utilized to optimize the hyperparameters of four models—MobileNetV2, MobileNetV3-Small, ShuffleNetV2,
and SqueezeNet—which are used for image classification. SMBO allows for efficient searching of the hyperparameter 
space by building a probabilistic model that predicts the best hyperparameters to test next.

Models
The following lightweight convolutional neural network models are explored for image classification:
1.	MobileNetV2: Efficient mobile-friendly architecture.
2.	MobileNetV3-Small: A smaller, optimized version for fast inference.
3.	ShuffleNetV2: An architecture designed for high computational efficiency.
4.	SqueezeNet: A model that achieves AlexNet-level accuracy with fewer parameters.
   
Optimization Approach
This project leverages Sequential Model-Based Optimization (SMBO), a powerful technique for hyperparameter tuning.
SMBO iteratively builds a surrogate model of the objective function (e.g., validation accuracy or loss) and uses it 
to select the next set of hyperparameters to try. This approach efficiently explores the hyperparameter space to find 
better configurations compared to traditional methods like grid or random search.

Tuned Hyperparameters:
•	Learning Rate: 0.0001-0.01.
•	Batch Size: 16,32,64,128,256.
•	Optimizer: Adam, RMSprop, and SGD .
•	Momentum: 0.0-0.99.

Training Strategy

The training process is divided into three parts:
1. Hyperparameter Optimization:
•	Goal: Find the best combination of hyperparameters using SMBO which reduce validation loss.
•	Configuration: The models are trained for 15 epochs with an early stopping patience of 3 epochs.
•	Trials: The optimization process runs for 300 trials, testing various combinations of hyperparameters.

3. Fitting the Model with Best Hyperparameters:
•	Once the optimal hyperparameters are found, the model is retrained using the best hyperparameter configuration.
•	Training Duration: The models are trained for 100 epochs with an early stopping patience of 10 epochs to ensure thorough training.

4. Testing the Model:
•	After training, the models are evaluated on a test set to determine their performance based on the optimal hyperparameters.
Dataset
The models were trained and validated on banana leaves and stem images dataset, which contains three classes (healthy, black sigatoka and fusarium wilt race 1).
 For more information related to dataset please read here: https://www.sciencedirect.com/science/article/pii/S2352340923004407

Results
After hyperparameter tuning, the models achieved the following results:

Model	Accuracy (%)	 Learning Rate	 Batch Size	Momentum	Optimizer	
MobileNetV2	97.73 	0.000016	128 	0.90 	SDG	
MobileNetV3-Small	96.33	0.000148	256 	0.966	RMSprop	
ShuffleNetV2	84.21 	0.000048	256 	0.9	SGD	
SqueezeNet	97.06	0.0000102	128 	0.9	SDG
	
These results demonstrate the effectiveness of SMBO in optimizing the models for improved performance on the image classification task.
License
This project is licensed under the MIT License - see the LICENSE file for details.

