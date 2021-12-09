
## SCORE DIFFICULTY ANALYSIS FOR PIANO PERFORMANCE EDUCATION

The paper will be available here. Submitted to ICASSP 2022.

## Abstract

In this paper, we introduce score difficulty classification as a subtask of music information retrieval (MIR),
which may be used in music education technologies, for personalised curriculum generation, and score retrieval. We
introduce a novel dataset for our task, Mikrokosmos-difficulty, containing 147 symbolic piano pieces and the 
corresponding difficulty labels derived by its composer Béla Bartók and the publishers. As part of our methodology,
we propose piano technique feature representations based on different piano fingering algorithms. We use these
features as input for two classifiers: a Gated Recurrent Unit neural network (GRU) with attention mechanism and 
gradient boosting trees trained on score segments. We show that for our dataset fingering based features perform better
than a simple baseline considering solely the notes in the score. Furthermore, the GRU with attention mechanism 
classifier surpasses the gradient boosting trees. Our proposed models are interpretable and are capable of generating
difficulty feedback both locally, on short term segments, and globally, for whole pieces. Code, datasets, models, and
an online demo are made available for reproducibility.

## Project Structure

- `approach_deepgru.py`: implementation deepgru method described on the paper.

- `approach_xgboost.py`: implementation xgboost method described on the paper.

- `export_feedback.py`: for generating difficulty feedback of both interpretable models.

- `loader_representations.py`: creates the five different representations.

- `mikrokosmos_fingering_pianoplayer.py`: fingering Mikrokosmos-difficulty dataset with Pianoplayer fynamic programming method.

- `mikrokosmos_fingering_nakamura.py`: fingering Mikrokosmos-difficulty dataset with Nakamura statistical method. 

- `table_dataset.py`: table 1.

- `tables.ipynb`: table 2, 3 and 4.

- Directory `results` the results and the final models.

- Directory `Fingers` the Nakamura and pianoplayer fingers in PIG format.

- Directory `representations` the results and the final models.

- Directory `mikrokosmos` the Mikrokosmos-difficulty dataset.


## Cite

If you use this your academic research, please cite the following:

```
@inproceedings{ramoneda2021difficulty,
  title={Score difficulty analysis for piano performance education},
  author={Ramoneda, Pedro and Tamer, Nazif Can and Eremenko, Vsevolod and Miron, Marius and Serra, Xavier},
  booktitle={Submitted to ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022}
}
```

