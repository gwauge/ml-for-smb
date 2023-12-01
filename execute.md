Execute in the background with 10min timeout:
```bash
nohup timeout 600 python src/leolm.py > output.log 2>&1 &
```

### View output
```bash
tail -f output.log
```
