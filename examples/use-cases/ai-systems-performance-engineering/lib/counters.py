"""EFA HTTP exporter and atomic Lustre textfile collector, using only the stdlib."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import time

EFA_FIELDS=('tx_bytes','rx_bytes','tx_pkts','rx_pkts','rdma_write_bytes','rdma_read_bytes',
            'rx_drops','retrans_bytes','retrans_timeout_events','unresponsive_remote_events')


def efa_metrics(root):
    lines=[]; devices=0
    for path in sorted(Path(root).glob('*/ports/*/hw_counters')):
        device=path.parents[2].name; port=path.parent.name
        labels=f'device={json.dumps(device)},port={json.dumps(port)}'
        if not (path/'tx_bytes').exists(): continue
        devices+=1
        for field in EFA_FIELDS:
            counter=path/field
            if counter.exists():
                try: value=int(counter.read_text().strip())
                except (OSError,ValueError): continue
                lines.append(f'node_amazonefa_{field}{{{labels}}} {value}')
    lines.append(f'aim347_efa_ports_visible {devices}')
    return '\n'.join(lines)+'\n'


def lustre_metrics(raw):
    # Keep the parameter names: omitting them with lctl -n loses mount identity.
    lines=[]; mount=None
    for line in raw.splitlines():
        if line.startswith('llite.') and line.endswith('.stats='):
            mount=line[len('llite.'):-len('.stats=')]
            continue
        fields=line.split()
        if mount is None or len(fields)<3 or fields[2]!='samples': continue
        key=fields[0]
        if key not in ('read_bytes','write_bytes','open','getattr'): continue
        count=int(fields[1])
        if key.endswith('_bytes'):
            # Lustre byte histograms are: samples [bytes] min max SUM [sum-squares].
            if len(fields)<7: continue
            value=int(fields[6]); metric='lustre_'+key+'_total'
        else:
            value=count; metric='lustre_'+key+'_operations_total'
        lines.append(f'{metric}{{mount={json.dumps(mount)}}} {value}')
    return '\n'.join(lines)+'\n'


def collect_lustre(destination):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        proc=subprocess.run(['lctl','get_param','llite.*.stats'],capture_output=True,text=True,check=True,timeout=5)
        data=lustre_metrics(proc.stdout)
        ok=bool(data.strip())
    except (OSError,subprocess.SubprocessError,ValueError):
        data=''; ok=False
    data+=f'aim347_lustre_collection_success {int(ok)}\naim347_lustre_collection_timestamp_seconds {time.time()}\n'
    with tempfile.NamedTemporaryFile('w',dir=destination.parent,delete=False,suffix='.tmp') as f:
        f.write(data); temporary=Path(f.name)
    temporary.chmod(0o644); temporary.replace(destination)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('efa'); e.add_argument('--root',default='/sys/class/infiniband'); e.add_argument('--port',type=int,default=9109)
    l=sub.add_parser('lustre'); l.add_argument('--output',required=True); l.add_argument('--once',action='store_true')
    a=p.parse_args()
    if a.mode=='lustre':
        while True:
            collect_lustre(a.output)
            if a.once: break
            time.sleep(5)
    else:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path!='/metrics': self.send_error(404); return
                data=efa_metrics(a.root).encode()
                self.send_response(200); self.send_header('Content-Type','text/plain; version=0.0.4'); self.end_headers(); self.wfile.write(data)
            def log_message(self,*args): pass
        HTTPServer(('0.0.0.0',a.port),Handler).serve_forever()


if __name__=='__main__': main()
