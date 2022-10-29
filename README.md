# Saliency Map for Neural Text Classifier
This repository is associated with the [Fake-EmoReact 2021 Challenge](https://sites.google.com/site/socialnlp2021/), the shared task of [SocialNLP@NAACL 2021](https://sites.google.com/site/socialnlp2021/).

The goal of the task is a binary classification problem detecting fake news tweets.
This repository contains the source code of the saliency map, visualizing the amount each token has contributed to the final prediction. See detailed description in *Section 7.* (Appendix) of [our report](https://drive.google.com/file/d/190NyQawdGIZ-scS1iLtuGBYZxeybCfXF/view?usp=share_link).

## The saliency map
![image](https://github.com/william0206/saliency-map/blob/main/fake_instance.png)
![image](https://github.com/william0206/saliency-map/blob/main/real_instance.png)

Check [this notebook](https://nbviewer.org/github/william0206/saliency-map/blob/main/saliency_map.ipynb) for the step-by-step saliency map implementation of the neural text classifier (*Bi-LSTM*) using PyTorch.

### Note
> If you want to re-run the whole notebook feel free to reach out and we could provide the dataset and model ckeckpoint.
