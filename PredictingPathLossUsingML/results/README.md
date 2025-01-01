Import these files into pandas using
```python
model_stats = pd.read_json(pipeline.DEFAULT_STATS_FILE, orient='records', lines=True)
print(model_stats.columns)
```