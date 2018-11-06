
## Automatic Emotion Recognition System for Laughter

This project uses machine learning to build an emotion recognition system capable of classifying various types of laughter, as well their dominance and arousal values. A practical reason for building a system capable of automatically interpreting the emotional content of laughter is to help a robot assistant understand all forms of communication that a human being might use around it.


## Dataset
The dataset used is the [MAHNOB Database](https://mahnob-db.eu/). In addition, more video clips were  collected by inducing subjects to laughter and the data was manually annotated.

## Results
[Report](https://github.com/oghabi/Emotion-Recognition-Classifier/blob/master/Report.pdf) discusses the video and audio data pre-processing methods and the feature extraction process, such as MFCC construction from the audio signals. It also compares and contrasts the performance of the various implemented machine learning and deep learning models on the dataset.

The best model was a weighted ensemble video and audio model which used VGG-16 to extract facial features from the video frames and LSTMs in order to exploit the temporal structure of video.

