# A Measure of Trustworthiness to Evaluate CNNs Predictions

Link to paper: https://arxiv.org/abs/2301.08839

**Abstract:** _Due to the black box nature of Convolutional neural networks (CNNs), the continuous validation of CNNs during operation is infeasible. As a result this makes it difficult for developers and regulators to gain confidence in the deployment of autonomous systems employing CNNs. It is critical for safety during operation to know when a CNN’s predictions are trustworthy or suspicious. The basic approach is to use the model’s output confidence score to assess if predictions are trustworthy or suspicious. However, the model’s confidence score is a result of computations coming from a black box, therefore lacks transparency and makes it challenging to credit trustworthiness to predictions. We introduce the trustworthiness score (TS), a simple metric that provides a more transparent and effective way of providing confidence in CNNs predictions. The metric quantifies the trustworthiness in a prediction by checking for the existence of certain features in the predictions made by the CNN. The TS metric can also be utilised to check for suspiciousness in a frame by scanning for the existence of untrustworthy predictions. We conduct a case study using YOLOv5 on persons detection to demonstrate our method and usage of the trustworthiness score. The case study shows that using our method consistently improves the precision of predictions compared to relying on model confidence alone, for both approving of trustworthy predictions (∼ 20% improvement) and detecting suspicious frames (∼ 5% improvement)._


Images below are examples highlighting the ground truth annotations (green boxes), main classifier object predictions (red boxes), and detected features specifications (magenta boxes) in the example images. The latter two are combined together in our method to output a trustworhtiness score in predictioins. 

![alt text](https://github.com/Abanoub-G/TrustworthinessScore/blob/main/paper_arxiv_submission/other/figures/INRIA_samples.png?raw=true)

![alt text](https://github.com/Abanoub-G/TrustworthinessScore/blob/main/paper_arxiv_submission/other/figures/COCO_samples1.png?raw=true)

