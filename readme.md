# Slurm Web
A simple website-based resource monitor for slurm system.

### Screenshot
<img src="slurm_web_example.png"
     width="60%"
     style="float: left; margin-right: 10px;" />

### Required python packages
`flask, colored, humanize, humanfriendly, beartype, seaborn, django`

### Launch
For example, run the command: `python app.py --host localhost --port 8080`.
Then the website will be hosted at `localhost:8080/`. 

You should change the host and port for your server.
Also change the [index.html](index.html) for header/footer and formatting.

### Reference
With some supports from [slurm_gpustat](https://github.com/albanie/slurm_gpustat).
