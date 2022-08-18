# Slurm Web
A simple website-based resource monitor for slurm system.

### Screenshot
<img src="slurm_web_example.png"
     width="60%"
     style="float: left; margin-right: 10px;" />

### Required python packages

Run 

`pip install -r requirements.txt`

to install the dependencies

### Launch
For example, run the command: `python app.py --host localhost --port 8080`.
Then the website will be hosted at `localhost:8080/`. 

You should change the host and port for your server.
Also change the [index.html](index.html) for header/footer and formatting.

### Running as a command-line tool

You can also use slurm_gpustat in the command line by

`python slurm_web/slurm_gpustat.py`

or by adding the following alias to your `.bash_profile`:

`alias slurm_gpustat='python ~/slurm_web/slurm_gpustat.py'`

### Reference
With some supports from [slurm_gpustat](https://github.com/albanie/slurm_gpustat).
