# Slurm web server
Host a website-based resource monitor for slurm system.

### Example
<img src="slurm_web_example.png"
     width="60%"
     style="float: left; margin-right: 10px;" />

### Required python packages
`flask, colored, humanize, humanfriendly, beartype, seaborn, django`

### Launch
run the command: `python app.py --host localhost --port 8080`

### Reference
With some supports from [slurm_gpustat](https://github.com/albanie/slurm_gpustat).
