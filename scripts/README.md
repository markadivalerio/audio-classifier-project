## Download Audioset CSVs to ./data/
# probably should just ignore unbal_train_segments.csv

```bash
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv

http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv

http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```

## Remove first two rows from eval_segments.csv and balanced_train_segments.csv

1.) Open up excel and delete the first two rows.
2.) Reaname the files to <same_name>-edited.csv

## Filter AudioSet to only animal sounds

1.) Open and run ../filter_audioset_data.ipynb

## Download wav files

```bash
cat balanced_train_segments-edited.csv | ./../scripts/download.sh

cat eval_segmentes-edited.csv | ./../scripts/download.sh
```