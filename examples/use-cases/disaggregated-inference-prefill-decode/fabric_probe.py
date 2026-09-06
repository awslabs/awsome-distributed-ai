"""Small NIXL GPU-buffer probe; a diagnostic helper, not an SGLang connector."""
import argparse
import base64
import json
import socket
import time
import uuid


def send(file, data):
    file.write(json.dumps(data).encode() + b'\n')
    file.flush()


def receive(file):
    line = file.readline(16 * 1024 * 1024)
    if not line:
        raise RuntimeError('Peer disconnected')
    return json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['server', 'client'])
    p.add_argument('--host', required=True, help='Server bind address or client destination on the private lab network')
    p.add_argument('--port', type=int, default=19000)
    p.add_argument('--bytes', type=int, default=64 * 1024 * 1024)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--backend', choices=['LIBFABRIC', 'UCX'], default='LIBFABRIC')
    p.add_argument('--eager', action='store_true', help='Call NIXL make_connection after metadata exchange, before timing transfers')
    p.add_argument('--instance-type', required=True)
    a = p.parse_args()
    assert a.bytes > 0 and a.iterations >= 2
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    name = 'aim345-' + uuid.uuid4().hex
    agent = nixl_agent(name, nixl_agent_config(backends=[a.backend]))
    tensor = torch.full((a.bytes,), 17 if a.role == 'client' else 0, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    registration = agent.register_memory(tensor)
    info = {'agent': name, 'metadata': base64.b64encode(agent.get_agent_metadata()).decode(), 'address': tensor.data_ptr(), 'bytes': a.bytes, 'gpu_index': tensor.get_device(), 'backend': a.backend}
    listener = None
    if a.role == 'server':
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host, a.port))
        listener.listen(1)
        listener.settimeout(120)
        print(json.dumps({'state': 'listening', 'port': a.port, 'instance_type': a.instance_type}), flush=True)
        conn, _ = listener.accept()
    else:
        conn = socket.create_connection((a.host, a.port), timeout=120)
    conn.settimeout(120)
    with conn, conn.makefile('rwb') as file:
        send(file, info)
        peer = receive(file)
        assert peer['bytes'] == a.bytes and peer['backend'] == a.backend
        agent.add_remote_agent(base64.b64decode(peer['metadata']))
        if a.eager:
            agent.make_connection(peer['agent'], backends=[a.backend])
        send(file, {'ready': True})
        assert receive(file)['ready']
        if a.role == 'client':
            local = agent.get_xfer_descs(tensor)
            remote = agent.get_xfer_descs([(peer['address'], a.bytes, peer['gpu_index'])], 'VRAM')
            handle = agent.initialize_xfer('WRITE', local, remote, peer['agent'], backends=[a.backend])
            elapsed = []
            for _ in range(a.iterations):
                start = time.perf_counter()
                agent.transfer(handle)
                while True:
                    state = agent.check_xfer_state(handle)
                    if state == 'DONE':
                        break
                    if state != 'PROC' or time.perf_counter() - start > 60:
                        raise RuntimeError(f'Transfer state {state}')
                    time.sleep(.0001)
                elapsed.append(time.perf_counter() - start)
            send(file, {'transferred': True})
            verification = receive(file)
            assert verification['correct'], verification
            print(json.dumps({'instance_type': a.instance_type, 'evidence_scope': 'mechanism-validation', 'backend': a.backend, 'buffer_memory': 'CUDA VRAM', 'gpu_count_per_peer': 1, 'bytes_per_transfer': a.bytes, 'iterations_count': a.iterations, 'eager_make_connection': a.eager, 'first_transfer_ms': elapsed[0] * 1000, 'warm_transfer_mean_ms': sum(elapsed[1:]) / len(elapsed[1:]) * 1000, 'warm_gib_per_s': a.bytes * len(elapsed[1:]) / sum(elapsed[1:]) / 1024**3, 'destination_verified': verification['correct'], 'qualification': 'Single-GPU buffer microbenchmark including Python polling; not SGLang KV latency or production bandwidth'}), flush=True)
            agent.release_xfer_handle(handle)
        else:
            assert receive(file)['transferred']
            torch.cuda.synchronize()
            correct = bool(torch.all(tensor == 17).item())
            send(file, {'correct': correct})
            assert correct
    agent.deregister_memory(registration)
    if listener:
        listener.close()

if __name__ == '__main__':
    main()
