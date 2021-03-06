# Accompanying code to the paper: X-AWARE: ConteXt-AWARE Human-Environment Attention Fusion for Driver Gaze Prediction in the Wild

X-AWARE uses GoCaRD! https://github.com/lstappen/GoCaRD

*Reliable systems for automatic estimation of the driver's gaze are crucial for reducing the number of traffic fatalities and for many emerging research areas aimed at developing intelligent vehicle-passenger systems. Gaze estimation is a challenging task, especially in environments with varying illumination and reflection properties. Furthermore, there is wide diversity with respect to the appearance of drivers' faces, both in terms of occlusions (e.g. vision aids) and cultural/ethnic backgrounds. For this reason, analysing the face along with contextual information -- for example, the vehicle cabin environment -- adds another, less subjective signal towards the design of robust systems for passenger gaze estimation.
In this paper, we present an integrated approach to jointly model different features for this task. In particular, to improve the fusion of the visually captured environment with the driver's face, we have developed a contextual attention mechanism, X-AWARE, attached directly to the output convolutional layers of InceptionResNetV networks. In order to showcase the effectiveness of our approach, we use the Driver Gaze in the Wild dataset, recently released as part of the Eighth Emotion Recognition in the Wild Challenge (EmotiW) challenge. Our best model outperforms the baseline by an absolute of 15.43% in accuracy on the validation set, and improves the previously best reported result by an absolute of 8.72% on the test set.*

## Citation
If you use these models, please cite the following paper:
```bibtex
@inproceedings{stappen2020x,
  title={X-AWARE: ConteXt-AWARE Human-Environment Attention Fusion for Driver Gaze Prediction in the Wild},
  author={Stappen, Lukas and Rizos, Georgios and Schuller, Bj{\"o}rn},
  booktitle={Proceedings of the 2020 International Conference on Multimodal Interaction},
  pages={858--867},
  year={2020}
}
```
