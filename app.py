import os
from datetime import datetime
import argparse
import copy
from flask import Flask, Response, render_template_string
from slurm_gpustat import resource_by_type, parse_all_gpus, gpu_usage, \
    node_states, INACCESSIBLE, parse_cmd, avail_stats_for_node

# from https://developer.nvidia.com/cuda-gpus
# sort gpu by computing power
CAPABILITY = {'a100': 8.0, 'a40': 8.6, 'a30': 8.0, 'a10': 8.6, 'a16': 8.6,
              'v100': 7.0, 'gv100gl': 7.0,
              'p40': 6.1, 'm40': 5.2,
              'rtx6k': 7.5, 'rtx8k': 7.5}


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
        total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
        user_summary = [f"{key}={val}" for key, val in sorted(subdict['n_gpu'].items(), 
                                                               key=lambda x: CAPABILITY.get(x[0], 10.0), 
                                                               reverse=True)]
        summary_str = ''.join([f'{i:12s}' for i in user_summary])
        out += f"{user:12s}[{total}]    {summary_str}\n"
    return out


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
                        f'{gpu_type}: {type_bar} gpus available</b></td></tr>')
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
        return render_template_string(open('index.html').read())

    @app.route('/time_feed')
    def time_feed():
        def generate():
            yield f'updated at: {datetime.now().strftime("%Y.%m.%d | %H:%M:%S")}'
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

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
