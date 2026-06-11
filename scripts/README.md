# Scripts folder

Put narration scripts here, organized by channel:

```
scripts/
  Demo/demo.csv          # starter example
  MyChannel/episode-01.csv
```

## CSV format

```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hello world.,001-intro,demo,0.5,0.5,0.8
```

- `voice` must match a key in `voices/presets.json`.
- `filename` is optional; auto-generated from text if omitted.

See `demo/demo.csv` for a starter script.
