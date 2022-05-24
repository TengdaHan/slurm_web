import os
from datetime import datetime
import argparse
import copy
import pytz
from collections import defaultdict
from subprocess import STDOUT, check_output
from flask import Flask, Response, render_template_string
from slurm_gpustat import resource_by_type, parse_all_gpus, gpu_usage, \
    node_states, INACCESSIBLE, parse_cmd, avail_stats_for_node, parse_node_names


# from https://developer.nvidia.com/cuda-gpus
# sort gpu by computing power
# if fail to display other GPU types, add items in the following dictionaries.
CAPABILITY = {'a100': 8.0, 'a40': 8.6, 'a30': 8.0, 'a10': 8.6, 'a16': 8.6,
              'v100': 7.0, 'gv100gl': 7.0,
              'p40': 6.1, 'm40': 5.2,
              'rtx6k': 7.5, 'rtx8k': 7.5}
GMEM = {'a6000': '[48g]', 'a40': '[48g]', 'a30': '[24g]',
        'v100': '[16g]', 'gv100gl': '[32g]',
        'p40': '[24g]', 'm40': '[12/24g]',
        'rtx6k': '[24g]', 'rtx8k': '[48g]'}
OLD_GPU_TYPES = ['p40', 'm40']


def get_resource_bar(avail, total, text='', long=False):
    """Create a long/short progress bar with text overlaid. Formatting handled in css."""

    if long:
        long_str = ' class=long'
    else:
        long_str = ''
    bar = (f'<div class="progress" data-text="{text}">'
           f'<progress{long_str} max="100" value="{avail/total*100}"></progress></div>')
    return bar


def str_to_int(text):
    """Convert strnig with unit to int. e.g. "30 GB" --> 30."""

    return int(''.join(c for c in text if c.isdigit()))


def parse_leaderboard():
    """Request sinfo, parse the leaderboard in string."""

    resources = parse_all_gpus()
    usage = gpu_usage(resources=resources, partition='gpu')
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]['n_gpu'] = {key: sum([x['n_gpu'] for x in val.values()])
                                     for key, val in subdict.items()}
        aggregates[user]['bash_gpu'] = {key: sum([x['bash_gpu'] for x in val.values()])
                                        for key, val in subdict.items()}
    out = ""
    for user, subdict in sorted(aggregates.items(),
                                key=lambda x: sum(x[1]['n_gpu'].values()), reverse=True):
        total = f"total={str(sum(subdict['n_gpu'].values())):2s}"
        user_summary = [f"{key}={val}" for key, val in sorted(subdict['n_gpu'].items(), 
                                                               key=lambda x: CAPABILITY.get(x[0], 10.0), 
                                                               reverse=True)]
        summary_str = ''.join([f'{i:12s}' for i in user_summary])
        num_new_gpus = [val for key, val in subdict['n_gpu'].items() if key not in OLD_GPU_TYPES]
        total += f"|newer={str(sum(num_new_gpus)):2s}"
        total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
        out += f"{user:12s}[{total}]    {summary_str}\n"
    return out


def cpu_usage(resources, partition='compute'):
    """Build a data structure of the CPU resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    cmd = "squeue -O NumNodes:100,nodelist:100,username:100,jobid:100 --noheader"
    if partition:
        cmd += f" --partition={partition}"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 4:
            continue
        cpu_count_str, node_str, user, jobid = tokens
        num_cpus = int(cpu_count_str.strip())
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            cpu_type = resources[node_name]["type"]

            if cpu_type in usage[user]:
                usage[user][cpu_type][node_name]['n_cpu'] += num_cpus
            else:
                usage[user][cpu_type] = defaultdict(lambda: {'n_cpu': 0,})
                usage[user][cpu_type][node_name]['n_cpu'] += num_cpus
    return usage


def parse_cpu_usage_to_table(partition='compute', show_bar=True):
    """Request sinfo for cnode, parse the output to a html table."""

    node_str = parse_cmd(f"sinfo -o '%1000N' --noheader --partition={partition}")
    assert isinstance(node_str, list) and len(node_str) == 1
    node_names = parse_node_names(node_str[0].strip())
    resources = {k:{'type': k[0:6]+'xx', 'count': 1} for k in node_names}
    states = node_states(partition=partition)
    res = {key: val for key, val in resources.items()
           if states.get(key, "down") not in INACCESSIBLE}
    res_total = copy.deepcopy(res)
    usage = cpu_usage(resources=res, partition=partition)

    for subdict in usage.values():
        for cpu_type, node_dicts in subdict.items():
            for node_name, user_cpu_count in node_dicts.items():
                count = res[node_name]["count"]
                count = max(count - user_cpu_count['n_cpu'], 0)
                res[node_name]["count"] = count

    res_total_by_type = defaultdict(list)
    for node, spec in res_total.items():
        res_total_by_type[spec["type"]].append({"node": node, "count": spec["count"]})
    
    res_usage_by_type = defaultdict(list)
    for node, spec in res.items():
        res_usage_by_type[spec["type"]].append({"node": node, "count": spec["count"]})

    table_html = []
    total_cpu_count = 0
    avail_cpu_count = 0

    # sort cpus from new to old
    type_list = sorted(list(res_total_by_type.keys()), 
                       reverse=True)

    # writing the html table
    for cpu_type in type_list:
        node_dicts = res_total_by_type[cpu_type]
        node_names = sorted([i['node'] for i in node_dicts])

        node_summaries = []
        num_col = []

        for node in node_names:
            node_name = f'<td>{node}</td>'

            users = [user for user in usage if node in usage[user].get(cpu_type, [])]
            if len(users):
                users = f"<td>user: {','.join(users)}</td>"
            else:
                users = f"<td>&nbsp</td>"
            
            detail_dict = avail_stats_for_node(node)
            detail_dict = {k: v for k, v in detail_dict.items() if k in ['cpu', 'mem']}

            if show_bar:
                c_stat = detail_dict['cpu'].split('/')
                c_stat = [int(i.strip()) for i in c_stat]
                cpu_bar = get_resource_bar(*c_stat, text=detail_dict['cpu'])

                m_stat = detail_dict['mem'].split('/')
                m_stat = [str_to_int(i.strip()) for i in m_stat]
                mem_bar = get_resource_bar(*m_stat, text=detail_dict['mem'], long=True)

                total_cpu_count += c_stat[-1]
                avail_cpu_count += c_stat[-1] - c_stat[0]
            else:
                cpu_bar = detail_dict['cpu']
                mem_bar = detail_dict['mem']
            cpu_stat = f"<td>cpu: {cpu_bar}</td>"
            mem_stat = f"<td>mem: {mem_bar}</td>"
            
            node_summary = f'<tr><td>&nbsp</td>{node_name}{cpu_stat}{mem_stat}{users}</tr>'
            node_summaries.append(node_summary)
            num_col.append(5)
            

        type_summary = (f'<tr><td colspan="{max(num_col)}"><b>'
                        f'{cpu_type}: </b></td></tr>')
        table_html.append(type_summary)
        table_html.extend(node_summaries)
        

    if show_bar:
        total_bar = get_resource_bar(avail_cpu_count, total_cpu_count, 
                                    text=f'{avail_cpu_count} / {total_cpu_count}')
        total_summary = (f'<tr><td colspan="{max(num_col)}"><h3>'
                     f'Summary: {total_bar} cpus available</h3></td></tr>')
    else:
        total_bar = f'{avail_cpu_count}/{total_cpu_count}'
        total_summary = ''
    
    table_html = f"<table>{total_summary}{''.join(table_html)}</table>"

    return table_html


def parse_usage_to_table(show_bar=True):
    """Request sinfo, parse the output to a html table."""

    resources = parse_all_gpus()
    states = node_states()
    res = {key: val for key, val in resources.items()
           if states.get(key, "down") not in INACCESSIBLE}
    res_total = copy.deepcopy(res)
    usage = gpu_usage(resources=res)

    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"] for x in res[node_name]].index(gpu_type)
                count = res[node_name][resource_idx]["count"]
                count = max(count - user_gpu_count['n_gpu'], 0)
                res[node_name][resource_idx]["count"] = count

    res_total_by_type = resource_by_type(res_total)
    res_usage_by_type = resource_by_type(res)

    table_html = []
    total_gpu_count = 0
    avail_gpu_count = 0

    # sort gpus from new to old
    type_list = sorted(list(res_total_by_type.keys()), 
                       key=lambda x: CAPABILITY.get(x, 10.0), 
                       reverse=True)

    # writing the html table
    for gpu_type in type_list:
        node_dicts = res_total_by_type[gpu_type]
        node_names = sorted([i['node'] for i in node_dicts])
        gpu_count_total = {i['node']:i['count'] for i in node_dicts}
        gpu_count_avail = {i['node']:i['count'] for i in res_usage_by_type[gpu_type]}

        node_summaries = []
        num_col = []
    
        for node in node_names:
            node_name = f'<td>{node}</td>'
            if show_bar:
                gpu_bar = get_resource_bar(gpu_count_avail[node], gpu_count_total[node], 
                                        text=f"{gpu_count_avail[node]} / {gpu_count_total[node]}") 
            else:
                gpu_bar = f'{gpu_count_avail[node]}/{gpu_count_total[node]}'
            gpu_stat = f'<td>gpu: {gpu_bar}</td>'
            
            users = [user for user in usage if node in usage[user].get(gpu_type, [])]
            if len(users):
                users = f"<td>user: {','.join(users)}</td>"
            else:
                users = f"<td>&nbsp</td>"
            
            detail_dict = avail_stats_for_node(node)
            detail_dict = {k: v for k, v in detail_dict.items() if k in ['cpu', 'mem']}

            if show_bar:
                c_stat = detail_dict['cpu'].split('/')
                c_stat = [int(i.strip()) for i in c_stat]
                cpu_bar = get_resource_bar(*c_stat, text=detail_dict['cpu'])

                m_stat = detail_dict['mem'].split('/')
                m_stat = [str_to_int(i.strip()) for i in m_stat]
                mem_bar = get_resource_bar(*m_stat, text=detail_dict['mem'], long=True)
            else:
                cpu_bar = detail_dict['cpu']
                mem_bar = detail_dict['mem']
            cpu_stat = f"<td>cpu: {cpu_bar}</td>"
            mem_stat = f"<td>mem: {mem_bar}</td>"
            
            node_summary = f'<tr><td>&nbsp</td>{node_name}{gpu_stat}{cpu_stat}{mem_stat}{users}</tr>'
            node_summaries.append(node_summary)
            num_col.append(6)
        
        if show_bar:
            type_bar = get_resource_bar(sum(gpu_count_avail.values()), sum(gpu_count_total.values()),
                                        text=f'{sum(gpu_count_avail.values())} / {sum(gpu_count_total.values())}')
        else:
            type_bar = f'{sum(gpu_count_avail.values())}/{sum(gpu_count_total.values())}'
        type_summary = (f'<tr><td colspan="{max(num_col)}"><b>'
                        f'{gpu_type} {GMEM.get(gpu_type, "")}: {type_bar} gpus available</b></td></tr>')
        table_html.append(type_summary)
        table_html.extend(node_summaries)
        total_gpu_count += sum(gpu_count_total.values())
        avail_gpu_count += sum(gpu_count_avail.values())

    if show_bar:
        total_bar = get_resource_bar(avail_gpu_count, total_gpu_count, 
                                    text=f'{avail_gpu_count} / {total_gpu_count}')
    else:
        total_bar = f'{avail_gpu_count}/{total_gpu_count}'
    total_summary = (f'<tr><td colspan="{max(num_col)}"><h3>'
                     f'Summary: {total_bar} gpus available</h3></td></tr>')
    table_html = f"<table>{total_summary}{''.join(table_html)}</table>"
    return table_html


def parse_queue_to_table():
    """Request pending queue, keep the raw formatting."""
    
    out = parse_cmd('squeue -t PENDING')
    out = '\n'.join(out)
    return out


def parse_disk_io():
    """Measure disk reading speed, parse the output to a html table.
    
    Pre-requisite: create a byte file by running 
    `dd if=/dev/zero of=/your/path/test.img bs=512MB count=1 oflag=dsync`."""

    try:
        beegfs_ultra_read = check_output(
            'dd if=/scratch/shared/beegfs/shared-datasets/test/test.img of=/dev/null bs=512MB count=1 oflag=dsync',
            stderr=STDOUT, shell=True).decode("utf-8")
        beegfs_ultra_read = beegfs_ultra_read.split('\n')[-2].split(',')[-1].strip()
    except:
        beegfs_ultra_read = 'N/A'

    try:
        beegfs_fast_read = check_output(
            'dd if=/scratch/shared/beegfs/htd/DATA/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync',
            stderr=STDOUT, shell=True).decode("utf-8") 
        beegfs_fast_read = beegfs_fast_read.split('\n')[-2].split(',')[-1].strip()
    except:
        beegfs_fast_read = 'N/A'

    try:
        beegfs_normal_read = check_output(
            'dd if=/scratch/shared/beegfs/htd/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync',
            stderr=STDOUT, shell=True).decode("utf-8") 
        beegfs_normal_read = beegfs_normal_read.split('\n')[-2].split(',')[-1].strip()
    except:
        beegfs_normal_read = 'N/A'

    try:
        work_normal_read = check_output(
            'dd if=/work/htd/Desktop_tmp/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync',
            stderr=STDOUT, shell=True).decode("utf-8") 
        work_normal_read = work_normal_read.split('\n')[-2].split(',')[-1].strip()
    except:
        work_normal_read = 'N/A'

    summary = '<tr> <td><b>{}</b></td> <td><b>{}</b></td> <td><b>{}</b></td> </tr>'.format(
        'Disk', 'Type', 'Read Speed')
    summary += '<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>'.format(
        '/beegfs/shared-datasets <i>[ultra-fast-layer]</i>', 'NVMe flash', beegfs_ultra_read)
    summary += '<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>'.format(
        '/beegfs <i>[fast-layer]</i>', 'SSD flash', beegfs_fast_read)
    summary += '<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>'.format(
        '/beegfs <i>[normal-layer]</i>', 'HDD', beegfs_normal_read)
    summary += '<tr> <td>{}</td> <td>{}</td> <td>{}</td></tr>'.format(
        '/work', 'HDD', work_normal_read)
    table_html = f"<table>{summary}</table>"

    return table_html


def main():
    parser = argparse.ArgumentParser(description="launch web app")
    parser.add_argument("--host", default='triton.robots.ox.ac.uk',
                        help="the host address for the website")
    parser.add_argument("--port", default=2070,
                        help="the port for the website")
    args = parser.parse_args()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(open('index.html').read(), hostname=args.host)

    @app.route('/time_feed')
    def time_feed():
        def generate():
            yield f'updated at: {datetime.now(pytz.timezone("Europe/London")).strftime("%Y.%m.%d | %H:%M:%S")}'
        return Response(generate(), mimetype='text')

    @app.route('/resource')
    def resource():
        def generate():
            out = parse_usage_to_table()
            yield out
        return Response(generate(), mimetype='text')

    @app.route('/queue')
    def queue():
        def generate():
            out = parse_queue_to_table()
            yield out
        return Response(generate(), mimetype='text')

    @app.route('/leaderboard')
    def leaderboard():
        def generate():
            out = parse_leaderboard()
            yield out
        return Response(generate(), mimetype='text')

    @app.route('/disk_io')
    def disk_io():
        def generate():
            out = parse_disk_io()
            yield out
        return Response(generate(), mimetype='text')

    # @app.route('/cpu_resource')
    # def cpu_resource():
    #     def generate():
    #         out = parse_cpu_usage_to_table()
    #         yield out
    #     return Response(generate(), mimetype='text')

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
