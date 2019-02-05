To resume training, you need to navigate to the log folder of the training job. 
By default, logs are saved in `/tmp/adept_logs`.
```bash
# Change directory to the desired log directory
cd /tmp/adept_logs/<environment-id>/<log-id-dir>/
# To continue training on a single GPU
python -m adept.app local --resume .
# To continue training on multiple GPUs
python -m adept.app distrib --resume .
```
